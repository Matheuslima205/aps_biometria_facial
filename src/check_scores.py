import sys
from pathlib import Path

import numpy as np

from features import cosine_similarity, extract
from pipeline import acquire, detect_face, preprocess

ROOT = Path(__file__).resolve().parent.parent

method = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
db = np.load(ROOT / "data" / f"db_{method}.npz")
names, templates = list(db["names"]), db["templates"]

for photo in sorted((ROOT / "data" / "test").glob("*.jpg")):
    _, gray = acquire(str(photo))
    roi, _ = detect_face(preprocess(gray))
    if roi is None:
        print(f"{photo.name:28s} nenhum rosto detectado")
        continue
    vec = extract(roi, method)
    scores = [cosine_similarity(vec, t) for t in templates]
    best = int(np.argmax(scores))
    print(f"{photo.name:28s} melhor={names[best]:20s} score={scores[best]:.3f}")