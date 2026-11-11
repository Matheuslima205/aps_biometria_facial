import sys
from pathlib import Path

import numpy as np

from features import extract
from pipeline import acquire, detect_face, preprocess

ROOT = Path(__file__).resolve().parent.parent
ENROLL_DIR = ROOT / "data" / "enroll"


def build_database(method="facenet"):
    """Rebuild the embedding database from the enrollment image folders."""
    names, templates = [], []
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)

    for user_dir in sorted(p for p in ENROLL_DIR.iterdir() if p.is_dir()):
        vectors = []
        photos = sorted(
            photo for photo in user_dir.iterdir()
            if photo.is_file() and photo.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for photo in photos:
            _, gray = acquire(str(photo))
            roi, _ = detect_face(preprocess(gray))
            if roi is None:
                print(f"    [PULANDO] Nenhum rosto detectado em {photo.name}")
                continue
            vectors.append(extract(roi, method))

        if not vectors:
            print(f"    [ALERTA] {user_dir.name}: nenhuma foto utilizável; usuário não registrado")
            continue

        template = np.mean(vectors, axis=0)
        norm = np.linalg.norm(template)
        if norm <= 1e-10:
            print(f"    [ALERTA] {user_dir.name}: embedding inválido; usuário não registrado")
            continue

        templates.append(template / norm)
        names.append(user_dir.name)
        print(f"{user_dir.name}: {len(vectors)} fotos registradas")

    out = ROOT / "data" / f"db_{method}.npz"
    if templates:
        np.savez(out, names=np.array(names), templates=np.array(templates))
        vector_size = len(templates[0])
    else:
        np.savez(
            out,
            names=np.array([], dtype=str),
            templates=np.empty((0, 0), dtype=np.float32),
        )
        vector_size = 0

    print(
        f"\nBanco de dados salvo: {out.name} ({len(names)} usuários, "
        f"tamanho do vetor {vector_size})"
    )
    return len(names)


if __name__ == "__main__":
    build_database(sys.argv[1] if len(sys.argv) > 1 else "dinov2")
