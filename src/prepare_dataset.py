import json
from pathlib import Path

import cv2
import numpy as np
from skimage import data as sk_data
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parent.parent
ENROLL_DIR = ROOT / "data" / "enroll"
TEST_DIR = ROOT / "data" / "test"

N_ENROLLED = 5 
N_UNKNOWN = 2 
ENROLL_PHOTOS = 5
TEST_PHOTOS = 2
ACCESS_LEVELS = [3, 2, 1, 1, 2]

def slug(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "_")

def save(img_rgb: np.ndarray, path: Path) -> None:
    img = img_rgb * 255.0 if img_rgb.max() <= 1.0 else img_rgb
    img = np.clip(img, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    

def main() -> None:
    print("Baixando/carregando o LFW (a primeira vez pode demorar um pouco)...")
    
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=1.0,
                           slice_=None, color=True)
    names = [str(n) for n in lfw.target_names]
    print(f"{len(lfw.images)} imagens de {len(names)} pessoas: {names}")
    
    rng = np.random.default_rng(42)
    people = rng.permutation(len(names))
    enrolled = people[:N_ENROLLED]
    unknown = people[N_ENROLLED:N_ENROLLED + N_UNKNOWN]
    
    users = {}
    for i, idx in enumerate(enrolled):
        user = slug(names[idx])
        photos = rng.permutation(np.where(lfw.target == idx)[0])
        for k, p in enumerate(photos[:ENROLL_PHOTOS], 1):
            save(lfw.images[p], ENROLL_DIR / user / f"{k}.jpg")
        for k, p in enumerate(photos[ENROLL_PHOTOS:ENROLL_PHOTOS + TEST_PHOTOS], 1):
            save(lfw.images[p], TEST_DIR / f"{user}_{k:02d}.jpg")
        users[user]= ACCESS_LEVELS[i % len(ACCESS_LEVELS)]
        
    n = 1 
    for idx in unknown:
        photos = rng.permutation(np.where(lfw.target == idx)[0])
        for p in photos[:TEST_PHOTOS]:
            save(lfw.images[p], TEST_DIR / f"unknown_{n:02d}.jpg")
            n += 1 
            
    invalid = [
        sk_data.coffee() / 255.0,
        sk_data.chelsea() / 255.0,
        np.full((250, 250, 3), 0.02),
        rng.random((250,250,3)),
    ]
    for k, img in enumerate(invalid, 1):
        save(img, TEST_DIR / f"invalid_{k:02d}.jpg")
        
    with open(ROOT / "users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
        
    print("\nUsuários aceitos (nome -> nível de acesso):")
    for user, level in users.items():
        print(f"    {user}: nível {level}")
    print(f"\n Imagens de teste: {len(list(TEST_DIR.glob('*.jpg')))}")
    print("Feito!")
    
if __name__ == "__main__":
    main()