export type Box = { x1: number; y1: number; x2: number; y2: number; score: number };

/** Traitements disponibles. */
export type Effect = 'faceghost' | 'pixelateme' | 'blackout';
export type Shape = 'ellipse' | 'rect';
export type Device = 'gpu' | 'cpu';

export interface ProcessOptions {
  effect: Effect;
  shape: Shape;
  model: string;
  device: Device;
  conf: number;
  iou: number;
  detectSize: number;
  detectEvery: number;
  padding: number;
  persist: number;
  strength: number;
  codec: string;
  quality: number;
  hwDecode: boolean;
  debugBoxes: boolean;
}

export interface VideoInfo {
  width: number;
  height: number;
  fps: string;
  fpsValue: number;
  frames: number;
  duration: number;
  audioCodec?: string;
}
