import ort from 'onnxruntime-node';
import type { Box, Device } from './types.js';

/** Détecteur de visages YOLO-face (ONNX) exécuté sur GPU via DirectML. */
export class YoloFaceDetector {
  private input?: Float32Array;
  private inputW = 0;
  private inputH = 0;

  private constructor(
    private session: ort.InferenceSession,
    private size: number,
    private conf: number,
    private iou: number,
  ) {}

  static async create(modelPath: string, device: Device, size: number, conf: number, iou: number) {
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: device === 'gpu' ? ['dml', 'cpu'] : ['cpu'],
      graphOptimizationLevel: 'all',
      enableMemPattern: device !== 'gpu',
      executionMode: 'sequential',
      logSeverityLevel: 3,
    });
    return new YoloFaceDetector(session, size, conf, iou);
  }

  async detect(frame: Buffer, width: number, height: number): Promise<Box[]> {
    // Redimensionnement en conservant le ratio, dimensions multiples de 32 (entrée dynamique).
    const scale = Math.min(this.size / width, this.size / height);
    const rw = Math.round(width * scale);
    const rh = Math.round(height * scale);
    const iw = Math.ceil(rw / 32) * 32;
    const ih = Math.ceil(rh / 32) * 32;
    if (iw !== this.inputW || ih !== this.inputH) {
      this.input = new Float32Array(3 * iw * ih);
      this.inputW = iw;
      this.inputH = ih;
    }
    const input = this.input!;
    input.fill(114 / 255);
    resizeBilinearToCHW(frame, width, height, input, iw, ih, rw, rh);

    const tensor = new ort.Tensor('float32', input, [1, 3, ih, iw]);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const result = out[this.session.outputNames[0]];
    const data = result.data as Float32Array;
    const n = result.dims[2];

    const boxes: Box[] = [];
    for (let i = 0; i < n; i++) {
      const score = data[4 * n + i];
      if (score < this.conf) continue;
      const cx = data[i] / scale;
      const cy = data[n + i] / scale;
      const w = data[2 * n + i] / scale;
      const h = data[3 * n + i] / scale;
      boxes.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score });
    }
    return nms(boxes, this.iou);
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}

function resizeBilinearToCHW(
  src: Buffer, sw: number, sh: number,
  dst: Float32Array, dw: number, dh: number, rw: number, rh: number,
): void {
  const plane = dw * dh;
  const xRatio = sw / rw;
  const yRatio = sh / rh;
  const inv = 1 / 255;
  for (let y = 0; y < rh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * yRatio - 0.5, 0), sh - 1);
    const y0 = fy | 0;
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = fy - y0;
    const row0 = y0 * sw * 3;
    const row1 = y1 * sw * 3;
    for (let x = 0; x < rw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * xRatio - 0.5, 0), sw - 1);
      const x0 = fx | 0;
      const x1 = Math.min(x0 + 1, sw - 1);
      const wx = fx - x0;
      const a = row0 + x0 * 3, b = row0 + x1 * 3, c = row1 + x0 * 3, d = row1 + x1 * 3;
      const o = y * dw + x;
      for (let ch = 0; ch < 3; ch++) {
        const top = src[a + ch] + (src[b + ch] - src[a + ch]) * wx;
        const bottom = src[c + ch] + (src[d + ch] - src[c + ch]) * wx;
        dst[ch * plane + o] = (top + (bottom - top) * wy) * inv;
      }
    }
  }
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

function nms(boxes: Box[], threshold: number): Box[] {
  boxes.sort((a, b) => b.score - a.score);
  const kept: Box[] = [];
  for (const box of boxes) {
    if (kept.every((k) => iou(k, box) < threshold)) kept.push(box);
  }
  return kept;
}
