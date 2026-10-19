export type Box = { x1: number; y1: number; x2: number; y2: number; score: number };

/** Traitements disponibles. */
export type Effect = 'faceghost' | 'pixelateme' | 'blackout' | 'personbox' | 'persontrack';
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
  /** Couleur RGB et épaisseur (px) des cadres pour personbox / --debug-boxes. */
  boxColor: [number, number, number];
  boxThickness: number;
  /** Personnes (personbox, persontrack) : suppression des faux positifs, voir TrackFilter. */
  minDuration: number;
  minMotion: number;
  minSpeed: number;
  maxAngle: number;
  /** persontrack : durée de la traînée en secondes (0 = trajectoire entière), cadres en plus des points. */
  trail: number;
  trackBoxes: boolean;
  /** Ligne de comptage verticale, position en fraction de la largeur (undefined = pas de comptage). */
  countLine?: number;
  /** Écrit les trajectoires retenues en JSON à côté de la vidéo. */
  exportTracks: boolean;
  /** Recadrage portrait 9:16 en 1080×1920 : position horizontale du cadre (0 = gauche, 0.5 = centre, 1 = droite). */
  portrait?: number;
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
