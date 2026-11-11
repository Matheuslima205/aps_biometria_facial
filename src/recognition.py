import json
from pathlib import Path

import cv2
import numpy as np

from features import cosine_similarity, extract
from pipeline import acquire, detect_face, preprocess

ROOT = Path(__file__).resolve().parent.parent
METHOD = "facenet"
THRESHOLD = 0.50

ACCESS_INFO = {
    1: "Nível 1 - Informações ambientais gerais: dados de propriedades rurais e consultas públicas.",
    2: "Nível 2 - Relatórios de fiscalização, ocorrências ambientais e análises técnicas.",
    3: "Nível 3 - Informações estratégicas, relatórios consolidados e decisões administrativas.",
}


def load_database(method=METHOD):
    db = np.load(ROOT / "data" / f"db_{method}.npz")
    with open(ROOT / "users.json", encoding="utf-8") as f:
        users = json.load(f)
    return list(db["names"]), db["templates"], users


def identify(path, method=METHOD, threshold=THRESHOLD):
    names, templates, users = load_database(method)
    img, gray = acquire(path)
    pre = preprocess(gray)
    roi, box = detect_face(pre)

    result = {"status": "invalid", "user": None, "score": None, "level": None,
              "original": img, "preprocessed": pre, "detection": None, "roi": None}
    if roi is None:
        return result

    x, y, w, h = box
    marked = img.copy()
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
    result["detection"] = marked
    result["roi"] = roi

    vec = extract(roi, method)
    scores = [cosine_similarity(vec, t) for t in templates]
    best = int(np.argmax(scores))
    result["score"] = scores[best]
    result["user"] = names[best]

    if scores[best] >= threshold:
        result["status"] = "recognized"
        result["level"] = users.get(names[best])
    else:
        result["status"] = "unknown"
    return result


def allowed_info(level):
    return [ACCESS_INFO[n] for n in range(1, level + 1)]


if __name__ == "__main__":
    import sys
    r = identify(sys.argv[1])
    print(r["status"], r["user"], r["score"], r["level"])
