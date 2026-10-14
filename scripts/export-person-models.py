"""Exporte les modèles YOLO11 COCO (détection de personnes) en ONNX dans models/.

Étape unique : le traitement vidéo reste 100 % TypeScript.

    python -m venv .venv-export
    .venv-export\\Scripts\\pip install ultralytics onnx onnxslim
    .venv-export\\Scripts\\python scripts/export-person-models.py          # n, s, m, l
    .venv-export\\Scripts\\python scripts/export-person-models.py m        # un seul
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

for size in sys.argv[1:] or ["n", "s", "m", "l"]:
    # Entrée dynamique : le détecteur envoie des images aux dimensions multiples de 32, pas forcément carrées.
    onnx = YOLO(f"yolo11{size}.pt").export(format="onnx", dynamic=True, simplify=True, opset=17)
    target = MODELS_DIR / f"yolo11{size}-person.onnx"
    shutil.move(onnx, target)
    print(f"✔ {target}")
