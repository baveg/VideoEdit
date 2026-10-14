import { spawn, execFile, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { once } from 'node:events';
import { promisify } from 'node:util';
import type { VideoInfo } from './types.js';

const execFileAsync = promisify(execFile);

/** Codecs audio qu'on peut copier tels quels dans un MP4. */
const MP4_AUDIO_COPY = new Set(['aac', 'mp3', 'ac3', 'eac3', 'opus', 'alac', 'flac']);

export async function probe(file: string): Promise<VideoInfo> {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', file,
  ]);
  const data = JSON.parse(stdout);
  const video = data.streams.find((s: any) => s.codec_type === 'video');
  if (!video) throw new Error('aucun flux vidéo trouvé');
  const audio = data.streams.find((s: any) => s.codec_type === 'audio');

  // FFmpeg applique automatiquement la rotation : on inverse largeur/hauteur si besoin.
  const rotation = Number(
    video.side_data_list?.find((d: any) => d.rotation !== undefined)?.rotation ?? video.tags?.rotate ?? 0,
  );
  const swap = Math.abs(rotation) % 180 === 90;
  const width = swap ? video.height : video.width;
  const height = swap ? video.width : video.height;

  const fps = video.avg_frame_rate && video.avg_frame_rate !== '0/0' ? video.avg_frame_rate : video.r_frame_rate;
  const [num, den] = fps.split('/').map(Number);
  const fpsValue = den ? num / den : num;
  const duration = Number(video.duration ?? data.format.duration ?? 0);
  const frames = Number(video.nb_frames) || Math.round(duration * fpsValue);

  return { width, height, fps, fpsValue, frames, duration, audioCodec: audio?.codec_name };
}

export function spawnDecoder(file: string, hwDecode: boolean): ChildProcessWithoutNullStreams {
  const args = ['-hide_banner', '-loglevel', 'error'];
  if (hwDecode) args.push('-hwaccel', 'cuda');
  args.push('-i', file, '-map', '0:v:0', '-fps_mode', 'cfr', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-');
  return spawn('ffmpeg', args);
}

export function spawnEncoder(
  input: string,
  output: string,
  info: VideoInfo,
  codec: string,
  quality: number,
): ChildProcessWithoutNullStreams {
  const args = [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${info.width}x${info.height}`, '-framerate', info.fps, '-i', '-',
    '-i', input,
    '-map', '0:v:0', '-map', '1:a?', '-map_metadata', '1',
    '-c:v', codec,
  ];
  if (codec.endsWith('_nvenc')) args.push('-preset', 'p5', '-rc', 'vbr', '-cq', String(quality), '-b:v', '0');
  else args.push('-preset', 'medium', '-crf', String(quality));
  args.push('-pix_fmt', 'yuv420p');
  args.push('-c:a', info.audioCodec && MP4_AUDIO_COPY.has(info.audioCodec) ? 'copy' : 'aac');
  args.push('-movflags', '+faststart', '-shortest', output);
  return spawn('ffmpeg', args);
}

/** Lit la sortie rawvideo du décodeur et produit une image complète à la fois. */
export async function* readFrames(proc: ChildProcessWithoutNullStreams, frameSize: number): AsyncGenerator<Buffer> {
  let pending: Buffer[] = [];
  let pendingSize = 0;
  for await (const chunk of proc.stdout as AsyncIterable<Buffer>) {
    pending.push(chunk);
    pendingSize += chunk.length;
    if (pendingSize < frameSize) continue;
    let buf = Buffer.concat(pending, pendingSize);
    let offset = 0;
    while (buf.length - offset >= frameSize) {
      yield buf.subarray(offset, offset + frameSize);
      offset += frameSize;
    }
    buf = buf.subarray(offset);
    pending = buf.length ? [Buffer.from(buf)] : [];
    pendingSize = buf.length;
  }
}

export async function writeFrame(proc: ChildProcessWithoutNullStreams, frame: Buffer): Promise<void> {
  if (!proc.stdin.write(frame)) await once(proc.stdin, 'drain');
}

/** Attend la fin d'un process ffmpeg et lève une erreur lisible en cas d'échec. */
export function waitExit(proc: ChildProcessWithoutNullStreams, label: string): Promise<void> {
  let stderr = '';
  proc.stderr.on('data', (d) => (stderr += d.toString()));
  return new Promise((resolve, reject) => {
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${label} a échoué (code ${code}) : ${stderr.trim().slice(-800)}`));
    });
  });
}

export async function hasEncoder(name: string): Promise<boolean> {
  try {
    const { stdout } = await execFileAsync('ffmpeg', ['-hide_banner', '-encoders']);
    return new RegExp(`\\s${name}\\s`).test(stdout);
  } catch {
    return false;
  }
}
