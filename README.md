# VideoEdit — anonymisation de visages (YOLO-face + FFmpeg, GPU)

Traite toutes les vidéos du dossier `input/` et écrit le résultat dans `output/`.
Tout est en TypeScript : aucune dépendance Python.

```
input/*.mp4 ─► FFmpeg (décodage CUDA) ─► YOLO-face ONNX (GPU via DirectML) ─► effet ─► FFmpeg (NVENC) ─► output/*_<mode>.mp4
                                                                                      └─ audio d'origine conservé
```

## Prérequis

- Node.js ≥ 20
- FFmpeg + ffprobe dans le PATH (build avec NVENC, ex. gyan.dev)
- Carte graphique compatible DirectX 12 (NVIDIA conseillée pour l'encodage NVENC)

```powershell
npm install
npm run doctor      # vérifie FFmpeg, NVENC, GPU et télécharge le modèle par défaut
```

## Utilisation

Déposez les vidéos dans `input/`, puis :

```powershell
npm run faceghost       # flou gaussien en ellipse (par défaut)
npm run pixelateme      # mosaïque
npm run blackout        # masque noir
npm run personbox       # cadre autour des personnes (caméra en hauteur, foule)
```

`personbox` n'anonymise pas : il détecte les personnes (YOLO11 COCO, classe « person ») et dessine un cadre autour.
Réglages par défaut adaptés à une caméra en hauteur : `--model yolo11m-person --detect-size 1280 --conf 0.35`
(personnes petites ou vues de dessus). Chaque option reste modifiable :

```powershell
npm run personbox -- --box-color "#ffcc00" --box-thickness 4
npm run personbox -- --model yolo11s-person --detect-size 960   # plus rapide
npm run personbox -- --model yolo11l-person                     # plus précis
```

Options (à ajouter après `--`) :

```powershell
npm run dev -- -i D:\rushes -o D:\anonymes -m pixelateme --strength 1.5
npm run dev -- --model yolov11s-face --detect-size 1280   # petits visages, vidéos 4K
npm run dev -- --debug-boxes -o output\debug              # affiche les détections
```

| Option | Défaut | Rôle |
|---|---|---|
| `-i, --input` | `input` | dossier source |
| `-o, --output` | `output` | dossier de sortie |
| `-m, --mode` | `faceghost` | `faceghost`, `pixelateme`, `blackout`, `personbox` |
| `--shape` | `ellipse` | `ellipse` ou `rect` |
| `--model` | `yolov8n-face` | nom de modèle (`npm run models`) ou chemin vers un `.onnx` |
| `--device` | `gpu` | `gpu` (DirectML) ou `cpu` |
| `--conf` | `0.3` | seuil de détection (baisser = masquer plus de visages) |
| `--detect-size` | `640` | résolution d'analyse ; `1280` pour les petits visages |
| `--detect-every` | `1` | détecter 1 image sur n (plus rapide) |
| `--padding` | `0.25` | marge autour du visage |
| `--persist` | `8` | images pendant lesquelles un visage perdu reste masqué |
| `--box-color` | `#00ff00` | couleur des cadres (`personbox`, `--debug-boxes`) |
| `--box-thickness` | auto | épaisseur des cadres en px |
| `--strength` | `1` | intensité du flou / taille des blocs |
| `--codec` | `h264_nvenc` | `hevc_nvenc`, `av1_nvenc`, `libx264`… |
| `--quality` | `21` | CQ (NVENC) ou CRF (x264) ; plus bas = meilleure qualité |
| `--no-hw-decode` | | désactive le décodage CUDA |
| `-j, --jobs` | `1` | vidéos traitées en parallèle |
| `--overwrite` | | retraite les fichiers déjà présents en sortie |
| `--external "<cmd>"` | | lance un script existant à la place (voir ci-dessous) |

Les vidéos déjà traitées sont ignorées, ce qui permet de relancer après une interruption.

## Modèles

```powershell
npm run models                  # liste (✔ = déjà téléchargé)
npm run models -- yolov8m-face  # télécharge
```

Les modèles de visages viennent de [deepghs/yolo-face](https://huggingface.co/deepghs/yolo-face) et sont téléchargés dans `models/` au premier usage.

Les modèles de personnes (`yolo11{n,s,m,l}-person`) ne sont pas publiés en ONNX : on les exporte une seule fois
depuis les poids officiels Ultralytics (le traitement reste sans Python) :

```powershell
python -m venv .venv-export
.venv-export\Scripts\pip install ultralytics onnx onnxslim
.venv-export\Scripts\python scripts/export-person-models.py
```

## Script externe (faceghost, PixelateMe, script Python…)

Si vous avez déjà un script, le projet sert simplement d'orchestrateur sur les dossiers :

```powershell
npm run dev -- --external "python C:\outils\faceghost.py --in {input} --out {output}"
```

`{input}` et `{output}` sont remplacés (entre guillemets) pour chaque vidéo.

## Version compilée

```powershell
npm run build
node dist/cli.js -m pixelateme
```

## Structure

```
src/cli.ts        commandes run / models / doctor
src/processor.ts  pipeline par vidéo + moteur externe
src/ffmpeg.ts     probe, décodeur/encodeur FFmpeg en pipes rawvideo
src/detector.ts   YOLO ONNX visages / personnes (prétraitement, NMS)
src/tracker.ts    lissage temporel des boîtes (anti-clignotement)
src/effects.ts    faceghost, pixelateme, blackout, cadres (personbox)
src/models.ts     catalogue et téléchargement des modèles
```
