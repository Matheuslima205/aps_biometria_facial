import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

HF_MODEL = "facebook/dinov2-small"


def extract_lbp(roi, points=8, radius=1, grid=4):
    lbp = local_binary_pattern(roi, points, radius, method="uniform")
    n_bins = points + 2
    h, w = lbp.shape
    hists = []
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * h // grid:(i + 1) * h // grid,
                       j * w // grid:(j + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
            hists.append(hist)
    return np.concatenate(hists).astype(np.float32)


class Dino2Extractor:
    def __init__(self, model_name=HF_MODEL):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        
    def __call__(self, roi):
        rgb = Image.fromarray(roi).convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs)
        return out.last_hidden_state[0, 0].numpy().astype(np.float32)
    
class FaceNetExtractor:
    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch = torch
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    def __call__(self, roi):
        rgb = np.stack([roi] * 3, axis=0).astype(np.float32)
        tensor = self.torch.from_numpy((rgb - 127.5) / 128.0).unsqueeze(0)
        with self.torch.no_grad():
            out = self.model(tensor)
        return out[0].numpy().astype(np.float32)


_dinov2 = None
_facenet = None

def extract(roi, method="dinov2"):
    global _dinov2, _facenet
    if method == "lbp":
        vec = extract_lbp(roi)
    elif method == "dinov2":
        if _dinov2 is None:
            _dinov2 = Dino2Extractor()
        vec = _dinov2(roi)
    elif method == "facenet":
        if _facenet is None:
            _facenet = FaceNetExtractor()
        vec = _facenet(roi)
    else:
        raise ValueError(f"Método desconhecido: {method}")
    return vec / (np.linalg.norm(vec) + 1e-10)

def cosine_similarity(a, b):
    return float(np.dot(a, b))