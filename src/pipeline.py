import glob
import cv2


def acquire(path, width=400):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Não foi encontrado: {path}")
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def preprocess(gray, blur_kernel=5, clip_limit=2.0):
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(blurred)


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(pre, roi_size=160):
    faces = FACE_CASCADE.detectMultiScale(
        pre,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60),
    )
    
    if len(faces) == 0:
        return None, None
    x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    roi = cv2.resize(pre[y:y + h, x:x + w], (roi_size, roi_size))
    return roi, (x, y, w, h)


if __name__ == "__main__":
    path = [p for p in sorted(glob.glob("data/test/*.jpg"))
        if "unknown_" not in p and "invalid_" not in p][0]
    print("Imagem:", path)
    img, gray = acquire(path) 
    pre = preprocess(gray)
    print("gray     -> min:", gray.min(), "max:", gray.max())
    print("pre      -> min:", pre.min(), "max:", pre.max())
    cv2.imwrite("outputs/1_original.jpg", img)
    cv2.imwrite("outputs/2_gray.jpg", gray)
    cv2.imwrite("outputs/3_preprocessed.jpg", pre)  
    
    roi, box = detect_face(pre)
    if roi is None:
        print("Nenhum rosto detectado")
    else:
        x, y, w, h = box
        print("Face box:", box)
        marked = img.copy()
        cv2.rectangle(marked, (x,y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("outputs/4_detection.jpg", marked)
        cv2.imwrite("outputs/5_roi.jpg", roi)