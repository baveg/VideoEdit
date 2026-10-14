import { spawn } from 'node:child_process';
import { YoloDetector } from './detector.js';
import { applyEffect, drawBoxes } from './effects.js';
import { probe, readFrames, spawnDecoder, spawnEncoder, waitExit, writeFrame } from './ffmpeg.js';
import { FaceTracker } from './tracker.js';
import type { Box, ProcessOptions } from './types.js';

export interface Progress {
  frame: number;
  total: number;
  fps: number;
  faces: number;
}

/** Floute/pixelise les visages (ou encadre les personnes) d'une vidéo : décodage FFmpeg → YOLO (GPU) → effet → encodage NVENC. */
export async function processVideo(
  detector: YoloDetector,
  input: string,
  output: string,
  opts: ProcessOptions,
  onProgress: (p: Progress) => void,
): Promise<void> {
  const info = await probe(input);
  const frameSize = info.width * info.height * 3;
  const tracker = new FaceTracker(opts.persist);
  const thickness = opts.boxThickness || Math.max(2, Math.round(Math.min(info.width, info.height) / 360));

  const decoder = spawnDecoder(input, opts.hwDecode);
  const encoder = spawnEncoder(input, output, info, opts.codec, opts.quality);
  const decoderDone = waitExit(decoder, 'décodage FFmpeg');
  const encoderDone = waitExit(encoder, 'encodage FFmpeg');
  // Si l'un des deux process échoue, on arrête l'autre pour ne pas bloquer.
  decoderDone.catch(() => encoder.kill());
  encoderDone.catch(() => decoder.kill());
  encoder.stdin.on('error', () => {});

  const start = Date.now();
  let frameIndex = 0;
  let boxes: Box[] = [];
  try {
    for await (const frame of readFrames(decoder, frameSize)) {
      if (frameIndex % opts.detectEvery === 0) {
        boxes = tracker.update(await detector.detect(frame, info.width, info.height));
      } else {
        boxes = tracker.current();
      }
      applyEffect(frame, info.width, info.height, boxes, opts.effect, opts.shape, opts.padding, opts.strength);
      if (opts.effect === 'personbox' || opts.debugBoxes) drawBoxes(frame, info.width, info.height, boxes, opts.boxColor, thickness);
      await writeFrame(encoder, frame);
      frameIndex++;
      if (frameIndex % 10 === 0) {
        onProgress({ frame: frameIndex, total: info.frames, fps: frameIndex / ((Date.now() - start) / 1000), faces: boxes.length });
      }
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
  onProgress({ frame: frameIndex, total: frameIndex, fps: frameIndex / ((Date.now() - start) / 1000), faces: boxes.length });
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
