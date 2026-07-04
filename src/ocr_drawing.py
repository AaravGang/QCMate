import cv2
import numpy as np
import pytesseract
import re
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

TESS_CONFIG = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
MIN_CONF    = 40


def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def _clean(text: str) -> str:
    """Fix common Tesseract misreads in CAD/technical fonts."""
    text = re.sub(r'\|', 'I', text)
    text = re.sub(r'\{\{|\{', '[', text)
    text = re.sub(r'\}\}|\}', ']', text)
    text = re.sub(r'(?<=[0-9])\s(?=[0-9])', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _get_region(x: float, y: float, w: int, h: int) -> str:
    if y > h * 0.88:
        return "title_block"
    elif x < w * 0.45 and y < h * 0.38:
        return "notes"           # top-left: general notes / BOM tables
    elif x > w * 0.6 and y < h * 0.12:
        return "revision_block"
    else:
        return "drawing"         # dimension annotations, views


def extract_text(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    processed = _preprocess(img)

    data = pytesseract.image_to_data(
        processed,
        config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    lines = defaultdict(list)
    for i in range(len(data['text'])):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        if conf >= MIN_CONF and text:
            key = (data['block_num'][i], data['line_num'][i])
            x   = data['left'][i] / 2
            y   = data['top'][i]  / 2
            lines[key].append((x, y, _clean(text)))

    result = defaultdict(list)
    for words in lines.values():
        words.sort(key=lambda w: w[0])
        line_text = ' '.join(w[2] for w in words)
        avg_x = sum(w[0] for w in words) / len(words)
        avg_y = sum(w[1] for w in words) / len(words)
        region = _get_region(avg_x, avg_y, w, h)
        result[region].append(line_text)

    return dict(result)

if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ds0_b26253f-1.png"
    results = extract_text(path)

    print(json.dumps(results, indent=2))
