import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { YoloDetector } from './detector.js';
import { applyEffect, drawBoxes, drawCounter, drawDot, drawLine, trackColor } from './effects.js';
import { probe, readFrames, spawnDecoder, spawnEncoder, waitExit, writeFrame } from './ffmpeg.js';
import { FaceTracker } from './tracker.js';
import { filterTracks, lineCrossings, PersonTracker, smoothTrack, type FilterReport, type Track } from './tracks.js';
import type { Box, ProcessOptions, VideoInfo } from './types.js';

export interface Progress {
  frame: number;
  total: number;
  fps: number;
  faces: number;
}

/** Bilan des modes personnes : filtrage des faux positifs et, avec --count-line, nombre de passages. */
export interface PersonReport extends FilterReport {
  crossings?: number;
}

/**
 * Traite une vidéo : décodage FFmpeg → YOLO (GPU) → effet → encodage NVENC.
 * Pour les modes personnes, retourne le bilan du filtrage et du comptage.
 */
export async function processVideo(
  detector: YoloDetector,
  input: string,
  output: string,
  opts: ProcessOptions,
  onProgress: (p: Progress) => void,
): Promise<PersonReport | undefined> {
  if (opts.effect === 'personbox' || opts.effect === 'persontrack') return processPersons(detector, input, output, opts, onProgress);

  const info = await probe(input);
  const tracker = new FaceTracker(opts.persist);
  const thickness = boxThickness(info, opts);
  const start = Date.now();
  let boxes: Box[] = [];
  const frames = await transcode(input, output, info, opts, async (frame, index) => {
    if (index % opts.detectEvery === 0) {
      boxes = tracker.update(await detector.detect(frame, info.width, info.height));
    } else {
      boxes = tracker.current();
    }
    applyEffect(frame, info.width, info.height, boxes, opts.effect, opts.shape, opts.padding, opts.strength);
    if (opts.debugBoxes) drawBoxes(frame, info.width, info.height, boxes, opts.boxColor, thickness);
    if ((index + 1) % 10 === 0) {
      onProgress({ frame: index + 1, total: info.frames, fps: (index + 1) / ((Date.now() - start) / 1000), faces: boxes.length });
    }
  });
  onProgress({ frame: frames, total: frames, fps: frames / ((Date.now() - start) / 1000), faces: boxes.length });
  return undefined;
}

/**
 * Personnes, en deux passes : 1) détection + suivi sur toute la vidéo, puis suppression des faux positifs
 * à partir des trajectoires complètes ; 2) ré-encodage avec les cadres (personbox) ou points + trajectoires (persontrack).
 */
async function processPersons(
  detector: YoloDetector,
  input: string,
  output: string,
  opts: ProcessOptions,
  onProgress: (p: Progress) => void,
): Promise<PersonReport> {
  const info = await probe(input);
  const total = info.frames * 2;
  const start = Date.now();
  const report = (frame: number, count: number) =>
    onProgress({ frame, total, fps: frame / ((Date.now() - start) / 1000), faces: count });

  // Passe 1 : détection et suivi.
  const tracker = new PersonTracker(opts.persist);
  const decoder = spawnDecoder(input, opts.hwDecode);
  const decoderDone = waitExit(decoder, 'décodage FFmpeg');
  let frames = 0;
  try {
    for await (const frame of readFrames(decoder, info.width * info.height * 3)) {
      if (frames % opts.detectEvery === 0) {
        const detections = await detector.detect(frame, info.width, info.height);
        tracker.update(frames, detections);
        if (frames % 10 === 0) report(frames, detections.length);
      }
      frames++;
    }
    await decoderDone;
  } catch (err) {
    decoder.kill();
    await decoderDone.catch(() => {});
    throw err;
  }

  const filtered = filterTracks(tracker.tracks(), {
    minFrames: Math.round(opts.minDuration * info.fpsValue),
    minMotion: opts.minMotion,
    minSpeed: opts.minSpeed,
    fps: info.fpsValue,
    maxAngle: opts.maxAngle,
  });
  const tracks = filtered.kept.map((t) => smoothTrack(t, Math.max(1, Math.round(info.fpsValue / 4))));
  if (opts.exportTracks) writeTracksJson(output, info, tracks);

  // Trajectoires présentes à chaque image (les points sont contigus après interpolation).
  const byFrame: Track[][] = Array.from({ length: frames }, () => []);
  for (const t of tracks) {
    for (const p of t.points) if (p.frame < frames) byFrame[p.frame].push(t);
  }

  // Comptage : instants où chaque personne franchit la ligne.
  const lineX = opts.countLine !== undefined ? opts.countLine * info.width : undefined;
  const crossings = lineX !== undefined ? lineCrossings(tracks, lineX) : [];
  let counted = 0;

  // Passe 2 : dessin et encodage.
  const thickness = boxThickness(info, opts);
  const trail = opts.trail > 0 ? Math.round(opts.trail * info.fpsValue) : Infinity;
  await transcode(input, output, info, opts, (frame, index) => {
    const visible = byFrame[index] ?? [];
    if (lineX !== undefined) {
      drawLine(frame, info.width, info.height, lineX, 0, lineX, info.height - 1, [0, 0, 0], thickness + 2);
      drawLine(frame, info.width, info.height, lineX, 0, lineX, info.height - 1, [255, 255, 255], thickness);
    }
    for (const t of visible) {
      const end = index - t.points[0].frame;
      const p = t.points[end];
      if (opts.effect === 'personbox') {
        drawBoxes(frame, info.width, info.height, [p.box], opts.boxColor, thickness);
        continue;
      }
      const color = trackColor(t.id);
      for (let i = Math.max(1, end - trail + 1); i <= end; i++) {
        const a = t.points[i - 1], b = t.points[i];
        drawLine(frame, info.width, info.height, a.x, a.y, b.x, b.y, color, thickness);
      }
      if (opts.trackBoxes) drawBoxes(frame, info.width, info.height, [p.box], color, Math.max(1, thickness >> 1));
      drawDot(frame, info.width, info.height, p.x, p.y, thickness * 2, color);
    }
    if (lineX !== undefined) {
      while (counted < crossings.length && crossings[counted] <= index) counted++;
      drawCounter(frame, info.width, info.height, `PASSAGES : ${counted}`);
    }
    if ((index + 1) % 10 === 0) report(frames + index + 1, visible.length);
  });
  report(total, 0);
  return lineX !== undefined ? { ...filtered, crossings: crossings.length } : filtered;
}

function boxThickness(info: VideoInfo, opts: ProcessOptions): number {
  return opts.boxThickness || Math.max(2, Math.round(Math.min(info.width, info.height) / 360));
}

/** Trajectoires retenues (positions lissées, en pixels) dans <sortie>.tracks.json. */
function writeTracksJson(output: string, info: VideoInfo, tracks: Track[]): void {
  const round = (v: number) => Math.round(v * 10) / 10;
  const data = {
    width: info.width,
    height: info.height,
    fps: info.fpsValue,
    tracks: tracks.map((t) => ({
      id: t.id,
      points: t.points.map((p) => ({
        frame: p.frame,
        x: round(p.x),
        y: round(p.y),
        box: [round(p.box.x1), round(p.box.y1), round(p.box.x2), round(p.box.y2)],
      })),
    })),
  };
  writeFileSync(output.replace(/\.[^.\\/]+$/, '.tracks.json'), JSON.stringify(data));
}

/** Décode la vidéo, laisse `perFrame` modifier chaque image RGB24 puis la ré-encode (audio d'origine conservé). */
async function transcode(
  input: string,
  output: string,
  info: VideoInfo,
  opts: ProcessOptions,
  perFrame: (frame: Buffer, index: number) => void | Promise<void>,
): Promise<number> {
  const decoder = spawnDecoder(input, opts.hwDecode);
  const encoder = spawnEncoder(input, output, info, opts.codec, opts.quality);
  const decoderDone = waitExit(decoder, 'décodage FFmpeg');
  const encoderDone = waitExit(encoder, 'encodage FFmpeg');
  // Si l'un des deux process échoue, on arrête l'autre pour ne pas bloquer.
  decoderDone.catch(() => encoder.kill());
  encoderDone.catch(() => decoder.kill());
  encoder.stdin.on('error', () => {});

  let frameIndex = 0;
  try {
    for await (const frame of readFrames(decoder, info.width * info.height * 3)) {
      await perFrame(frame, frameIndex);
      await writeFrame(encoder, frame);
      frameIndex++;
    }
    encoder.stdin.end();
    await Promise.all([decoderDone, encoderDone]);
  } catch (err) {
    decoder.kill();
    encoder.kill();
    // Remonte l'erreur FFmpeg la plus parlante si disponible.
    await Promise.allSettled([decoderDone, encoderDone]).then((r) => {
      const failed = r.find((x) => x.status === 'rejected') as PromiseRejectedResult | undefined;
      if (failed) throw failed.reason;
    });
    throw err;
  }
  return frameIndex;
}

/**
 * Moteur externe : lance un script existant (faceghost, PixelateMe, script Python…)
 * avec les placeholders {input} et {output}.
 */
export function runExternal(template: string, input: string, output: string): Promise<void> {
  const command = template.replaceAll('{input}', `"${input}"`).replaceAll('{output}', `"${output}"`);
  return new Promise((resolve, reject) => {
    const proc = spawn(command, { shell: true, stdio: 'inherit' });
    proc.on('error', reject);
    proc.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`commande externe en échec (code ${code}) : ${command}`))));
  });
}
