import { existsSync, mkdirSync, createWriteStream, renameSync } from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';

export const MODELS_DIR = path.resolve('models');

/**
 * Modèles YOLO en ONNX : YOLO-face (sortie [1, 5, N]) et YOLO11 COCO pour les personnes (sortie [1, 84, N]).
 * Les modèles sans `url` ne sont pas hébergés en ONNX : ils s'exportent une fois avec scripts/export-person-models.py.
 */
export const MODELS: Record<string, { url?: string; note: string }> = {
  'yolov8n-face': { url: hf('yolov8n-face'), note: 'très rapide (12 Mo) — par défaut' },
  'yolov8m-face': { url: hf('yolov8m-face'), note: 'plus précis, plus lent' },
  'yolov8l-face': { url: hf('yolov8l-face'), note: 'le plus précis des v8' },
  'yolov11n-face': { url: hf('yolov11n-face'), note: 'YOLO11 nano' },
  'yolov11s-face': { url: hf('yolov11s-face'), note: 'YOLO11 small — bon compromis' },
  'yolov11m-face': { url: hf('yolov11m-face'), note: 'YOLO11 medium' },
  'yolo11n-person': { note: 'personnes — YOLO11 nano, très rapide' },
  'yolo11s-person': { note: 'personnes — YOLO11 small' },
  'yolo11m-person': { note: 'personnes — YOLO11 medium, par défaut pour personbox' },
  'yolo11l-person': { note: 'personnes — YOLO11 large, le plus précis' },
};

function hf(name: string): string {
  return `https://huggingface.co/deepghs/yolo-face/resolve/main/${name}/model.onnx`;
}

/** Retourne le chemin du modèle (nom connu ou chemin .onnx), en le téléchargeant si besoin. */
export async function resolveModel(nameOrPath: string): Promise<string> {
  if (nameOrPath.endsWith('.onnx')) {
    if (!existsSync(nameOrPath)) throw new Error(`modèle introuvable : ${nameOrPath}`);
    return path.resolve(nameOrPath);
  }
  const model = MODELS[nameOrPath];
  if (!model) throw new Error(`modèle inconnu "${nameOrPath}". Disponibles : ${Object.keys(MODELS).join(', ')}`);
  const file = path.join(MODELS_DIR, `${nameOrPath}.onnx`);
  if (!existsSync(file)) {
    if (!model.url) throw new Error(`modèle ${nameOrPath} absent de ${MODELS_DIR} : lancez "python scripts/export-person-models.py"`);
    await download(model.url, file);
  }
  return file;
}

async function download(url: string, file: string): Promise<void> {
  mkdirSync(path.dirname(file), { recursive: true });
  console.log(`Téléchargement du modèle ${path.basename(file)}…`);
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`téléchargement impossible (${res.status}) : ${url}`);
  const tmp = `${file}.part`;
  await pipeline(Readable.fromWeb(res.body as any), createWriteStream(tmp));
  renameSync(tmp, file);
}
