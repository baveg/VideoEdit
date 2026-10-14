import { iou } from './detector.js';
import type { Box } from './types.js';

/** Position d'une personne sur une image : centre de la boîte (x, y) et boîte détectée. */
export interface TrackPoint {
  frame: number;
  x: number;
  y: number;
  box: Box;
}

export interface Track {
  id: number;
  points: TrackPoint[];
}

interface LiveTrack extends Track {
  box: Box;
  vx: number;
  vy: number;
  missed: number;
}

/**
 * Suivi multi-personnes : chaque détection est associée à la trajectoire dont la position
 * prédite (vitesse constante) la recouvre le plus. Les trajectoires terminées sont conservées
 * pour le filtrage hors-ligne (voir filterTracks).
 */
export class PersonTracker {
  private live: LiveTrack[] = [];
  private finished: Track[] = [];
  private nextId = 1;

  /**
   * @param minIou recouvrement minimal entre position prédite et détection
   * @param maxDistance à défaut de recouvrement (déplacement rapide, vidéo accélérée, flou de bougé) :
   *   distance maximale entre les centres, en tailles de boîte
   */
  constructor(private persist: number, private minIou = 0.15, private maxDistance = 0.8) {}

  update(frame: number, detections: Box[]): void {
    // Associations gloutonnes par score décroissant : d'abord le recouvrement (IoU),
    // puis la proximité des centres pour les boîtes de taille comparable.
    const pairs: [number, number, number][] = [];
    this.live.forEach((t, ti) => {
      const dt = frame - t.points[t.points.length - 1].frame;
      const predicted = shift(t.box, t.vx * dt, t.vy * dt);
      const pc = center(predicted);
      const size = Math.sqrt((predicted.x2 - predicted.x1) * (predicted.y2 - predicted.y1));
      detections.forEach((d, di) => {
        const v = iou(predicted, d);
        if (v >= this.minIou) {
          pairs.push([v, ti, di]);
          return;
        }
        const ratio = (d.y2 - d.y1) / (predicted.y2 - predicted.y1);
        if (ratio < 0.6 || ratio > 1.6) return;
        const dc = center(d);
        const dist = Math.hypot(dc.x - pc.x, dc.y - pc.y) / size;
        if (dist < this.maxDistance) pairs.push([this.minIou * (1 - dist / this.maxDistance), ti, di]);
      });
    });
    pairs.sort((a, b) => b[0] - a[0]);
    const usedTracks = new Set<number>();
    const usedDets = new Set<number>();
    for (const [, ti, di] of pairs) {
      if (usedTracks.has(ti) || usedDets.has(di)) continue;
      usedTracks.add(ti);
      usedDets.add(di);
      this.extend(this.live[ti], frame, detections[di]);
    }

    for (const [ti, t] of this.live.entries()) if (!usedTracks.has(ti)) t.missed++;
    for (const t of this.live.filter((t) => t.missed > this.persist)) this.finished.push(strip(t));
    this.live = this.live.filter((t) => t.missed <= this.persist);

    for (const [di, d] of detections.entries()) {
      if (usedDets.has(di)) continue;
      const c = center(d);
      this.live.push({ id: this.nextId++, points: [{ frame, ...c, box: d }], box: d, vx: 0, vy: 0, missed: 0 });
    }
  }

  /** Toutes les trajectoires (terminées et en cours). */
  tracks(): Track[] {
    return [...this.finished, ...this.live.map(strip)];
  }

  private extend(t: LiveTrack, frame: number, det: Box): void {
    const last = t.points[t.points.length - 1];
    const c = center(det);
    const dt = frame - last.frame;
    // Vitesse lissée (la boîte détectée tremble), initialisée dès la deuxième position.
    const a = t.points.length === 1 ? 1 : 0.4;
    t.vx = (1 - a) * t.vx + a * ((c.x - last.x) / dt);
    t.vy = (1 - a) * t.vy + a * ((c.y - last.y) / dt);
    t.box = det;
    t.missed = 0;
    t.points.push({ frame, ...c, box: det });
  }
}

/** Paramètres de suppression des faux positifs. */
export interface TrackFilter {
  /** Durée minimale d'une trajectoire, en images. */
  minFrames: number;
  /** Déplacement net minimal, en hauteurs de personne (0 = garde les personnes immobiles). */
  minMotion: number;
  /**
   * Vitesse nette minimale, en hauteurs de personne par seconde : écarte les personnes à l'arrêt
   * dont la boîte dérive lentement (spectateurs, objets), sans pénaliser les personnes lointaines.
   */
  minSpeed: number;
  fps: number;
  /** Écart maximal (degrés) avec le sens de déplacement des voisins (180 = pas de filtre). */
  maxAngle: number;
}

export interface FilterReport {
  kept: Track[];
  short: number;
  static: number;
  wrongWay: number;
}

/**
 * Supprime les faux positifs à partir des trajectoires complètes :
 * détections trop brèves, personnes ou objets immobiles (spectateurs, drapeau…) et mouvements à contre-sens du cortège.
 * Le sens de référence est local (moyenne des voisins) car l'objectif grand angle courbe les trajectoires.
 */
export function filterTracks(tracks: Track[], f: TrackFilter): FilterReport {
  const report: FilterReport = { kept: [], short: 0, static: 0, wrongWay: 0 };
  const long = tracks.filter((t) => {
    const ok = t.points[t.points.length - 1].frame - t.points[0].frame + 1 >= f.minFrames;
    if (!ok) report.short++;
    return ok;
  });

  const stats = long.map((t) => {
    const a = t.points[0], b = t.points[t.points.length - 1];
    const height = median(t.points.map((p) => p.box.y2 - p.box.y1));
    const dx = b.x - a.x, dy = b.y - a.y;
    const motion = Math.hypot(dx, dy) / height;
    const speed = (motion * f.fps) / (b.frame - a.frame + 1);
    const mx = mean(t.points.map((p) => p.x)), my = mean(t.points.map((p) => p.y));
    return { t, dx, dy, motion, speed, mx, my, height };
  });

  const moving = stats.filter((s) => {
    const ok = s.motion >= f.minMotion && s.speed >= f.minSpeed;
    if (!ok) report.static++;
    return ok;
  });

  const cosMax = Math.cos((f.maxAngle * Math.PI) / 180);
  for (const s of moving) {
    if (f.maxAngle < 180) {
      // Sens moyen des voisins (pondéré par la proximité), hors trajectoire courante.
      let sx = 0, sy = 0;
      for (const o of moving) {
        if (o === s || o.motion < 0.5) continue;
        const d = Math.hypot(o.mx - s.mx, o.my - s.my) / s.height;
        const w = Math.exp(-d / 6);
        const n = Math.hypot(o.dx, o.dy);
        sx += (w * o.dx) / n;
        sy += (w * o.dy) / n;
      }
      const ns = Math.hypot(sx, sy), nt = Math.hypot(s.dx, s.dy);
      if (ns > 0.5 && nt > 0 && (s.dx * sx + s.dy * sy) / (ns * nt) < cosMax) {
        report.wrongWay++;
        continue;
      }
    }
    report.kept.push(s.t);
  }
  return report;
}

/**
 * Prépare une trajectoire pour l'affichage : comble les images sans détection par interpolation
 * linéaire et lisse les positions (moyenne glissante) pour un tracé régulier.
 */
export function smoothTrack(t: Track, radius = 4): Track {
  const filled: TrackPoint[] = [];
  for (let i = 0; i < t.points.length; i++) {
    const p = t.points[i];
    const next = t.points[i + 1];
    filled.push(p);
    if (!next) continue;
    for (let f = p.frame + 1; f < next.frame; f++) {
      const a = (f - p.frame) / (next.frame - p.frame);
      filled.push({ frame: f, x: lerp(p.x, next.x, a), y: lerp(p.y, next.y, a), box: lerpBox(p.box, next.box, a) });
    }
  }
  const points = filled.map((p, i) => {
    const lo = Math.max(0, i - radius), hi = Math.min(filled.length - 1, i + radius);
    let x = 0, y = 0;
    for (let j = lo; j <= hi; j++) { x += filled[j].x; y += filled[j].y; }
    const n = hi - lo + 1;
    return { ...p, x: x / n, y: y / n };
  });
  return { id: t.id, points };
}

/**
 * Images où chaque trajectoire franchit la verticale x = lineX (une seule fois par personne,
 * pour ne pas compter plusieurs fois quelqu'un qui piétine sur la ligne). Triées par ordre chronologique.
 */
export function lineCrossings(tracks: Track[], lineX: number): number[] {
  const frames: number[] = [];
  for (const t of tracks) {
    for (let i = 1; i < t.points.length; i++) {
      const a = t.points[i - 1].x - lineX, b = t.points[i].x - lineX;
      if ((a < 0 && b >= 0) || (a >= 0 && b < 0)) {
        frames.push(t.points[i].frame);
        break;
      }
    }
  }
  return frames.sort((x, y) => x - y);
}

function center(b: Box): { x: number; y: number } {
  return { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 };
}

function shift(b: Box, dx: number, dy: number): Box {
  return { x1: b.x1 + dx, y1: b.y1 + dy, x2: b.x2 + dx, y2: b.y2 + dy, score: b.score };
}

function strip(t: LiveTrack): Track {
  return { id: t.id, points: t.points };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpBox(a: Box, b: Box, t: number): Box {
  return { x1: lerp(a.x1, b.x1, t), y1: lerp(a.y1, b.y1, t), x2: lerp(a.x2, b.x2, t), y2: lerp(a.y2, b.y2, t), score: lerp(a.score, b.score, t) };
}

function mean(v: number[]): number {
  return v.reduce((s, x) => s + x, 0) / v.length;
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1];
}
