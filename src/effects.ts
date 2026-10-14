import type { Box, Effect, Shape } from './types.js';

interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Applique l'effet choisi sur chaque visage, directement dans l'image RGB24. */
export function applyEffect(
  frame: Buffer, width: number, height: number,
  boxes: Box[], effect: Effect, shape: Shape, padding: number, strength: number,
): void {
  for (const box of boxes) {
    const region = padRegion(box, padding, width, height);
    if (region.w < 2 || region.h < 2) continue;
    if (effect === 'faceghost') ghost(frame, width, region, shape, strength);
    else if (effect === 'pixelateme') pixelate(frame, width, region, shape, strength);
    else fill(frame, width, region, shape);
  }
}

export function drawBoxes(frame: Buffer, width: number, height: number, boxes: Box[]): void {
  for (const b of boxes) {
    const x1 = clamp(Math.round(b.x1), 0, width - 1), x2 = clamp(Math.round(b.x2), 0, width - 1);
    const y1 = clamp(Math.round(b.y1), 0, height - 1), y2 = clamp(Math.round(b.y2), 0, height - 1);
    for (let t = 0; t < 2; t++) {
      for (let x = x1; x <= x2; x++) { setPx(frame, width, x, Math.min(y1 + t, y2), 0, 255, 0); setPx(frame, width, x, Math.max(y2 - t, y1), 0, 255, 0); }
      for (let y = y1; y <= y2; y++) { setPx(frame, width, Math.min(x1 + t, x2), y, 0, 255, 0); setPx(frame, width, Math.max(x2 - t, x1), y, 0, 255, 0); }
    }
  }
}

function padRegion(b: Box, padding: number, width: number, height: number): Region {
  const pw = (b.x2 - b.x1) * padding;
  const ph = (b.y2 - b.y1) * padding;
  const x1 = clamp(Math.floor(b.x1 - pw), 0, width);
  const y1 = clamp(Math.floor(b.y1 - ph * 1.2), 0, height); // un peu plus haut pour couvrir les cheveux
  const x2 = clamp(Math.ceil(b.x2 + pw), 0, width);
  const y2 = clamp(Math.ceil(b.y2 + ph), 0, height);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/** Poids du masque (0..1) pour un pixel de la région ; `feather` adoucit le bord de l'ellipse. */
function maskWeight(px: number, py: number, r: Region, shape: Shape, feather: boolean): number {
  if (shape === 'rect') return 1;
  const dx = (px + 0.5 - r.w / 2) / (r.w / 2);
  const dy = (py + 0.5 - r.h / 2) / (r.h / 2);
  const d = dx * dx + dy * dy;
  if (!feather) return d <= 1 ? 1 : 0;
  if (d <= 0.7) return 1;
  if (d >= 1) return 0;
  return (1 - d) / 0.3;
}

/**
 * faceghost : flou gaussien (approximé par 3 flous boîte) fondu dans une ellipse.
 * Le flou est calculé sur une copie réduite de la zone puis ré-agrandi en bilinéaire :
 * visuellement identique pour un flou fort, et bien plus rapide sur les grands visages.
 */
function ghost(frame: Buffer, width: number, r: Region, shape: Shape, strength: number): void {
  const radius = Math.max(3, Math.round((Math.max(r.w, r.h) / 6) * strength));
  const f = Math.max(1, Math.floor(radius / 4));
  const sw = Math.ceil(r.w / f);
  const sh = Math.ceil(r.h / f);

  // Réduction par moyenne de blocs f×f.
  const small = new Float32Array(sw * sh * 3);
  for (let sy = 0; sy < sh; sy++) {
    for (let sx = 0; sx < sw; sx++) {
      const x0 = sx * f, y0 = sy * f;
      const x1 = Math.min(x0 + f, r.w), y1 = Math.min(y0 + f, r.h);
      let cr = 0, cg = 0, cb = 0;
      for (let y = y0; y < y1; y++) {
        let o = ((r.y + y) * width + r.x + x0) * 3;
        for (let x = x0; x < x1; x++, o += 3) { cr += frame[o]; cg += frame[o + 1]; cb += frame[o + 2]; }
      }
      const n = (x1 - x0) * (y1 - y0);
      const s = (sy * sw + sx) * 3;
      small[s] = cr / n; small[s + 1] = cg / n; small[s + 2] = cb / n;
    }
  }

  const sr = Math.max(1, Math.round(radius / f));
  const tmp = new Float32Array(small.length);
  for (let pass = 0; pass < 3; pass++) {
    boxBlur(small, tmp, sw, sh, sr, true);
    boxBlur(tmp, small, sw, sh, sr, false);
  }

  for (let y = 0; y < r.h; y++) {
    const fy = Math.min(Math.max((y + 0.5) / f - 0.5, 0), sh - 1);
    const y0 = fy | 0, y1 = Math.min(y0 + 1, sh - 1), wy = fy - y0;
    for (let x = 0; x < r.w; x++) {
      const a = maskWeight(x, y, r, shape, true);
      if (a === 0) continue;
      const fx = Math.min(Math.max((x + 0.5) / f - 0.5, 0), sw - 1);
      const x0 = fx | 0, x1 = Math.min(x0 + 1, sw - 1), wx = fx - x0;
      const p00 = (y0 * sw + x0) * 3, p01 = (y0 * sw + x1) * 3, p10 = (y1 * sw + x0) * 3, p11 = (y1 * sw + x1) * 3;
      const o = ((r.y + y) * width + r.x + x) * 3;
      for (let c = 0; c < 3; c++) {
        const top = small[p00 + c] + (small[p01 + c] - small[p00 + c]) * wx;
        const bottom = small[p10 + c] + (small[p11 + c] - small[p10 + c]) * wx;
        const v = top + (bottom - top) * wy;
        frame[o + c] = frame[o + c] + (v - frame[o + c]) * a;
      }
    }
  }
}

/** Flou boîte séparable (fenêtre glissante, bords étendus). */
function boxBlur(src: Float32Array, dst: Float32Array, w: number, h: number, radius: number, horizontal: boolean): void {
  const len = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const step = horizontal ? 3 : w * 3;
  const last = len - 1;
  const inv = 1 / (radius * 2 + 1);
  for (let line = 0; line < lines; line++) {
    const base = horizontal ? line * w * 3 : line * 3;
    for (let c = 0; c < 3; c++) {
      const b = base + c;
      let sum = 0;
      for (let i = -radius; i <= radius; i++) sum += src[b + (i < 0 ? 0 : i > last ? last : i) * step];
      for (let i = 0; i < len; i++) {
        dst[b + i * step] = sum * inv;
        const add = i + radius + 1;
        const sub = i - radius;
        sum += src[b + (add > last ? last : add) * step] - src[b + (sub < 0 ? 0 : sub) * step];
      }
    }
  }
}

/** pixelateme : mosaïque, chaque bloc prend la couleur moyenne de ses pixels. */
function pixelate(frame: Buffer, width: number, r: Region, shape: Shape, strength: number): void {
  const block = Math.max(4, Math.round((Math.max(r.w, r.h) / 10) * strength));
  for (let by = 0; by < r.h; by += block) {
    for (let bx = 0; bx < r.w; bx += block) {
      const bw = Math.min(block, r.w - bx);
      const bh = Math.min(block, r.h - by);
      let sr = 0, sg = 0, sb = 0;
      for (let y = 0; y < bh; y++) {
        let o = ((r.y + by + y) * width + r.x + bx) * 3;
        for (let x = 0; x < bw; x++, o += 3) { sr += frame[o]; sg += frame[o + 1]; sb += frame[o + 2]; }
      }
      const n = bw * bh;
      sr /= n; sg /= n; sb /= n;
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++) {
          if (!maskWeight(bx + x, by + y, r, shape, false)) continue;
          setPx(frame, width, r.x + bx + x, r.y + by + y, sr, sg, sb);
        }
      }
    }
  }
}

/** blackout : masque plein noir. */
function fill(frame: Buffer, width: number, r: Region, shape: Shape): void {
  for (let y = 0; y < r.h; y++) {
    for (let x = 0; x < r.w; x++) {
      if (maskWeight(x, y, r, shape, false)) setPx(frame, width, r.x + x, r.y + y, 0, 0, 0);
    }
  }
}

function setPx(frame: Buffer, width: number, x: number, y: number, r: number, g: number, b: number): void {
  const o = (y * width + x) * 3;
  frame[o] = r;
  frame[o + 1] = g;
  frame[o + 2] = b;
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}
