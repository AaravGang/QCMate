import cv2
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled from {w}x{h} to {gray.shape[1]}x{gray.shape[0]}")

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("outputs/gray.png",     gray)
    cv2.imwrite("outputs/enhanced.png", enhanced)
    cv2.imwrite("outputs/adaptive.png", adaptive)
    cv2.imwrite("outputs/closed.png",   closed)

    print("Saved: outputs/gray.png, enhanced.png, adaptive.png, closed.png")
    return closed


if __name__ == "__main__":
    result = preprocess("data/raw/ds0_A4-1798-1.png")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False, drop_score=0.3)
        ocr_result = ocr.ocr(result)
        print("\n=== OCR on preprocessed image ===")
        if ocr_result and ocr_result[0]:
            for block in ocr_result[0]:
                if isinstance(block[1], (tuple, list)):
                    text, conf = block[1][0], block[1][1]
                    if len(text.strip()) >= 2 and conf >= 0.5:
                        print(f"[{conf:.2f}] {text}")
    except ImportError:
        print("PaddleOCR not available — preprocessed images saved to outputs/")
