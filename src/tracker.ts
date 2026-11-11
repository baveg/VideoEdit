import { iou } from './detector.js';
import type { Box } from './types.js';

interface Track {
  box: Box;
  missed: number;
  /** A dépassé la taille max au moins une fois : suivi mais plus masqué. */
  ignored: boolean;
}

/**
 * Lissage temporel : un visage reste masqué quelques images après une détection manquée,
 * ce qui évite les « flashs » où un visage redevient visible.
 * Une détection plus haute que `maxHeight` (costume, gros plan) rend sa piste définitivement ignorée :
 * le masque ne clignote pas quand la taille oscille autour du seuil.
 */
export class FaceTracker {
  private tracks: Track[] = [];

  constructor(private persist: number, private maxHeight = Infinity, private smoothing = 0.6) {}

  update(detections: Box[]): Box[] {
    const unmatched = new Set(detections);
    for (const track of this.tracks) {
      let best: Box | undefined;
      let bestIou = 0.2;
      for (const det of unmatched) {
        const v = iou(track.box, det);
        if (v > bestIou) [best, bestIou] = [det, v];
      }
      if (best) {
        unmatched.delete(best);
        const a = this.smoothing;
        track.box = {
          x1: a * best.x1 + (1 - a) * track.box.x1,
          y1: a * best.y1 + (1 - a) * track.box.y1,
          x2: a * best.x2 + (1 - a) * track.box.x2,
          y2: a * best.y2 + (1 - a) * track.box.y2,
          score: best.score,
        };
        track.missed = 0;
        if (this.tooBig(best)) track.ignored = true;
      } else {
        track.missed++;
      }
    }
    this.tracks = this.tracks.filter((t) => t.missed <= this.persist);
    for (const det of unmatched) this.tracks.push({ box: det, missed: 0, ignored: this.tooBig(det) });
    return this.current();
  }

  current(): Box[] {
    return this.tracks.filter((t) => !t.ignored).map((t) => t.box);
  }

  private tooBig(b: Box): boolean {
    return b.y2 - b.y1 > this.maxHeight;
  }
}
