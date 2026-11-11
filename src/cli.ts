#!/usr/bin/env node
import { Command, Option } from 'commander';
import { existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import ort from 'onnxruntime-node';
import { YoloDetector } from './detector.js';
import { hasEncoder } from './ffmpeg.js';
import { MODELS, resolveModel } from './models.js';
import { processVideo, runExternal, type Progress } from './processor.js';
import type { Device, Effect, ProcessOptions, Shape } from './types.js';

const VIDEO_EXT = new Set(['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.wmv', '.mts', '.m2ts', '.flv', '.mpg', '.mpeg']);

/** Commander passe la valeur par défaut en 2e argument : avec parseInt, elle servirait de base (radix). */
const toInt = (v: string) => parseInt(v, 10);

/**
 * Réglages des modes personnes (caméra en hauteur : personnes petites, vues de dessus), sauf si précisés.
 * Seuil de confiance bas : les faux positifs sont ensuite éliminés par le filtrage des trajectoires.
 */
const PERSON_DEFAULTS = { model: 'yolo11m-person', detectSize: 1280, conf: 0.3, persist: 15 };
const PERSON_MODES = new Set(['personbox', 'persontrack']);

const program = new Command()
  .name('videoedit')
  .description('Anonymisation de visages et détection de personnes dans des vidéos (YOLO + FFmpeg, accéléré GPU)');

program
  .command('run', { isDefault: true })
  .description('traite toutes les vidéos du dossier d\'entrée')
  .option('-i, --input <dir>', 'dossier des vidéos à traiter', 'input')
  .option('-o, --output <dir>', 'dossier de sortie', 'output')
  .addOption(new Option('-m, --mode <effect>', 'traitement à appliquer').choices(['faceghost', 'pixelateme', 'blackout', 'personbox', 'persontrack']).default('faceghost'))
  .addOption(new Option('--shape <shape>', 'forme du masque').choices(['ellipse', 'rect']).default('ellipse'))
  .option('--model <name|file.onnx>', `modèle YOLO (${Object.keys(MODELS).join(', ')})`, 'yolov8n-face')
  .addOption(new Option('--device <device>', 'exécution du détecteur').choices(['gpu', 'cpu']).default('gpu'))
  .option('--conf <n>', 'seuil de confiance de détection', parseFloat, 0.3)
  .option('--iou <n>', 'seuil NMS', parseFloat, 0.45)
  .option('--detect-size <px>', 'taille d\'analyse (1280 pour les petits visages / 4K)', toInt, 640)
  .option('--detect-every <n>', 'détecter 1 image sur n (plus rapide, moins précis)', toInt, 1)
  .option('--padding <ratio>', 'marge autour du visage', parseFloat, 0.25)
  .option('--max-face <ratio>', 'ignore les visages plus hauts que cette fraction de l\'image (0 = pas de limite)', parseFloat, 0)
  .option('--persist <frames>', 'garde le masque n images après la perte d\'un visage', toInt, 8)
  .option('--strength <n>', 'intensité du flou / taille des blocs (1 = normal)', parseFloat, 1)
  .option('--codec <codec>', 'encodeur vidéo (h264_nvenc, hevc_nvenc, av1_nvenc, libx264)', 'h264_nvenc')
  .option('--quality <n>', 'qualité CQ/CRF (plus bas = meilleur)', toInt, 21)
  .option('--no-hw-decode', 'désactive le décodage CUDA')
  .option('--debug-boxes', 'dessine les boîtes détectées (vérification)')
  .option('--box-color <hex>', 'couleur des cadres (personbox, --debug-boxes)', '#00ff00')
  .option('--box-thickness <px>', 'épaisseur des cadres et traits (défaut : selon la résolution)', toInt)
  .option('--min-duration <s>', 'personnes : ignore les détections présentes moins de n secondes', parseFloat, 0.5)
  .option('--min-motion <n>', 'personnes : déplacement minimal en hauteurs de personne (0 = garde les immobiles)', parseFloat, 0.5)
  .option('--min-speed <n>', 'personnes : vitesse minimale en hauteurs de personne par seconde (0 = garde les immobiles)', parseFloat, 0.15)
  .option('--max-angle <deg>', 'personnes : écart max avec le sens des voisins (180 = garde les contre-sens)', parseFloat, 90)
  .option('--trail <s>', 'persontrack : longueur de la trajectoire en secondes (0 = entière)', parseFloat, 0)
  .option('--track-boxes', 'persontrack : dessine aussi les cadres')
  .option('--count-line [ratio]', 'personnes : ligne verticale de comptage (position 0–1, 0.5 = centre) et compteur de passages', parseFloat)
  .option('--export-tracks', 'personnes : écrit les trajectoires dans <vidéo>.tracks.json')
  .option('--portrait [x]', 'recadre en portrait 9:16 (1080×1920), x = position horizontale 0–1 (0.5 = centre)', parseFloat)
  .option('--logo [file]', 'incruste un logo en haut à droite (défaut : <dossier d\'entrée>/logo.png)')
  .option('--logo-size <ratio>', 'largeur du logo en fraction de la largeur de la vidéo', parseFloat, 0.15)
  .option('-j, --jobs <n>', 'nombre de vidéos traitées en parallèle', toInt, 1)
  .option('--overwrite', 'retraite les vidéos déjà présentes en sortie')
  .option('--external <command>', 'utilise un script externe à la place, ex: "python faceghost.py {input} {output}"')
  .action(run);

program
  .command('models')
  .description('liste ou télécharge les modèles YOLO-face')
  .argument('[names...]', 'modèles à télécharger')
  .action(async (names: string[]) => {
    if (!names.length) {
      for (const [name, m] of Object.entries(MODELS)) {
        const ok = existsSync(path.join('models', `${name}.onnx`)) ? '✔' : ' ';
        console.log(` ${ok} ${name.padEnd(15)} ${m.note}`);
      }
      return;
    }
    for (const name of names) console.log(`✔ ${await resolveModel(name)}`);
  });

program
  .command('doctor')
  .description('vérifie FFmpeg, NVENC et le GPU')
  .action(doctor);

program.parseAsync().catch((err) => {
  console.error(`\n✖ ${err instanceof Error ? err.message : err}`);
  process.exit(1);
});

async function run(o: any, cmd: Command): Promise<void> {
  if (PERSON_MODES.has(o.mode)) {
    for (const [key, value] of Object.entries(PERSON_DEFAULTS)) {
      if (cmd.getOptionValueSource(key) === 'default') o[key] = value;
    }
  }
  const inputDir = path.resolve(o.input);
  const outputDir = path.resolve(o.output);
  if (!existsSync(inputDir)) throw new Error(`dossier d'entrée introuvable : ${inputDir}`);
  mkdirSync(outputDir, { recursive: true });

  const videos = readdirSync(inputDir)
    .filter((f) => VIDEO_EXT.has(path.extname(f).toLowerCase()) && statSync(path.join(inputDir, f)).isFile())
    .sort();
  if (!videos.length) {
    console.log(`Aucune vidéo dans ${inputDir}`);
    return;
  }

  const opts: ProcessOptions = {
    effect: o.mode as Effect,
    shape: o.shape as Shape,
    model: o.model,
    device: o.device as Device,
    conf: o.conf,
    iou: o.iou,
    detectSize: o.detectSize,
    detectEvery: Math.max(1, o.detectEvery),
    padding: o.padding,
    maxFace: o.maxFace,
    persist: o.persist,
    strength: o.strength,
    codec: o.codec,
    quality: o.quality,
    hwDecode: o.hwDecode,
    debugBoxes: !!o.debugBoxes,
    boxColor: parseHexColor(o.boxColor),
    boxThickness: o.boxThickness ?? 0,
    minDuration: o.minDuration,
    minMotion: o.minMotion,
    minSpeed: o.minSpeed,
    maxAngle: o.maxAngle,
    trail: o.trail,
    trackBoxes: !!o.trackBoxes,
    // --count-line sans valeur : commander donne true → ligne au centre.
    countLine: o.countLine === undefined ? undefined : o.countLine === true ? 0.5 : o.countLine,
    exportTracks: !!o.exportTracks,
    // --portrait sans valeur : cadre centré.
    portrait: o.portrait === undefined ? undefined : o.portrait === true ? 0.5 : o.portrait,
    // --logo sans valeur : logo.png du dossier d'entrée.
    logo: o.logo === undefined ? undefined : path.resolve(o.logo === true ? path.join(inputDir, 'logo.png') : o.logo),
    logoSize: o.logoSize,
  };
  if (opts.logo && !existsSync(opts.logo)) throw new Error(`logo introuvable : ${opts.logo}`);

  if (!o.external && !(await hasEncoder(opts.codec))) {
    console.warn(`⚠ encodeur ${opts.codec} indisponible, bascule sur libx264`);
    opts.codec = 'libx264';
  }

  const tasks = videos
    .map((file) => ({
      file,
      input: path.join(inputDir, file),
      output: path.join(outputDir, `${path.parse(file).name}_${o.external ? 'external' : opts.effect}${opts.portrait !== undefined ? '_portrait' : ''}${opts.logo ? '_logo' : ''}.mp4`),
    }))
    .filter((t) => {
      if (!o.overwrite && existsSync(t.output)) {
        console.log(`↷ ${t.file} déjà traité (utilisez --overwrite)`);
        return false;
      }
      return true;
    });

  const modelPath = o.external ? '' : await resolveModel(opts.model);
  console.log(
    o.external
      ? `Moteur externe : ${o.external}`
      : `Traitement : ${opts.effect} | modèle ${path.basename(modelPath)} sur ${opts.device.toUpperCase()} | encodeur ${opts.codec}`,
  );
  console.log(`${tasks.length} vidéo(s) à traiter → ${outputDir}\n`);

  const failures: string[] = [];
  const queue = [...tasks];
  const jobs = Math.max(1, Math.min(o.jobs, tasks.length));
  const startAll = Date.now();

  await Promise.all(
    Array.from({ length: jobs }, async () => {
      const detector = o.external
        ? undefined
        : await YoloDetector.create(modelPath, opts.device, opts.detectSize, opts.conf, opts.iou);
      try {
        for (let task = queue.shift(); task; task = queue.shift()) {
          const t0 = Date.now();
          try {
            if (o.external) await runExternal(o.external, task.input, task.output);
            const report = o.external
              ? undefined
              : await processVideo(detector!, task.input, task.output, opts, (p) => progress(task.file, p, jobs));
            clearLine(jobs);
            console.log(`✔ ${task.file} → ${path.basename(task.output)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
            if (report) {
              console.log(
                `  ${report.kept.length} personne(s) suivie(s) — écartées : ${report.short} trop brèves, ` +
                  `${report.static} immobiles, ${report.wrongWay} à contre-sens`,
              );
              if (report.crossings !== undefined) console.log(`  ${report.crossings} passage(s) de la ligne de comptage`);
            }
          } catch (err) {
            clearLine(jobs);
            console.error(`✖ ${task.file} : ${err instanceof Error ? err.message : err}`);
            failures.push(task.file);
            if (existsSync(task.output)) unlinkSync(task.output);
          }
        }
      } finally {
        await detector?.release();
      }
    }),
  );

  console.log(`\nTerminé en ${((Date.now() - startAll) / 1000).toFixed(1)} s — ${tasks.length - failures.length} ok, ${failures.length} en échec`);
  if (failures.length) process.exitCode = 1;
}

function progress(file: string, p: Progress, jobs: number): void {
  const pct = p.total ? Math.min(100, (p.frame / p.total) * 100) : 0;
  const eta = p.fps > 0 && p.total ? Math.max(0, (p.total - p.frame) / p.fps) : 0;
  const bar = '█'.repeat(Math.round(pct / 5)).padEnd(20, '░');
  const line = `${bar} ${pct.toFixed(0).padStart(3)}% ${file} | ${p.fps.toFixed(1)} img/s | ${p.faces} détection(s) | reste ${eta.toFixed(0)} s`;
  // En parallèle, les lignes se mélangeraient : on n'affiche la progression qu'en mode séquentiel.
  if (jobs === 1 && process.stdout.isTTY) process.stdout.write(`\r${line.slice(0, process.stdout.columns - 1)}`);
}

function parseHexColor(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`couleur invalide "${hex}" (format attendu : #rrggbb)`);
  const v = parseInt(m[1], 16);
  return [v >> 16, (v >> 8) & 255, v & 255];
}

function clearLine(jobs: number): void {
  if (jobs === 1 && process.stdout.isTTY) process.stdout.write('\r\x1b[K');
}

async function doctor(): Promise<void> {
  const check = (ok: boolean, label: string) => console.log(`${ok ? '✔' : '✖'} ${label}`);
  const { execFileSync } = await import('node:child_process');
  try {
    const v = execFileSync('ffmpeg', ['-version']).toString().split('\n')[0];
    check(true, v);
  } catch {
    check(false, 'ffmpeg introuvable dans le PATH');
  }
  for (const enc of ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc']) check(await hasEncoder(enc), `encodeur ${enc}`);
  try {
    const gpu = execFileSync('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader']).toString().trim();
    check(true, `GPU : ${gpu}`);
  } catch {
    check(false, 'nvidia-smi introuvable');
  }
  try {
    const model = await resolveModel('yolov8n-face');
    const det = await YoloDetector.create(model, 'gpu', 640, 0.3, 0.45);
    const frame = Buffer.alloc(1920 * 1080 * 3);
    await det.detect(frame, 1920, 1080);
    const t0 = performance.now();
    for (let i = 0; i < 20; i++) await det.detect(frame, 1920, 1080);
    check(true, `ONNX Runtime ${ort.env.versions.common} DirectML : ${((performance.now() - t0) / 20).toFixed(1)} ms/image (1080p)`);
    await det.release();
  } catch (err) {
    check(false, `détecteur GPU : ${err instanceof Error ? err.message : err}`);
  }
}
