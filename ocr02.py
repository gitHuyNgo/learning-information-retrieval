import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

FRAMES_PATH = Path("frames.jsonl")
OUT_PATH = Path("frames_gated.jsonl")

# ---- knobs (tune) ----
BLUR_THR = 100.0          # variance of Laplacian
DIFF_THR = 0.012          # ratio of changed pixels in ROI
ROI = (0.15, 0.10, 0.70, 0.75)  
# (x, y, w, h) theo tỉ lệ khung hình: crop vùng trung tâm (thường là board)

RESIZE_W = 320            # downscale ROI để tính diff ổn định & nhanh

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def crop_roi(img: np.ndarray, roi: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = img.shape[:2]
    rx, ry, rw, rh = roi
    x1 = int(rx * w)
    y1 = int(ry * h)
    x2 = int((rx + rw) * w)
    y2 = int((ry + rh) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]

def blur_score(gray: np.ndarray) -> float:
    # variance of Laplacian: higher = sharper
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def prep_for_diff(gray: np.ndarray) -> np.ndarray:
    # resize + blur nhẹ để giảm noise
    h, w = gray.shape[:2]
    if w != RESIZE_W:
        new_h = int(h * (RESIZE_W / w))
        gray = cv2.resize(gray, (RESIZE_W, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def diff_ratio(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, cur_gray)
    # threshold diff để lấy pixel thay đổi
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return float(np.mean(mask > 0))

def main():
    rows = read_jsonl(FRAMES_PATH)
    prev = None

    kept = 0
    for i, r in enumerate(rows):
        path = r["path"]
        img = cv2.imread(path)
        if img is None:
            r["do_ocr"] = False
            r["gating_scores"] = {"error": "imread_failed"}
            continue

        roi_img = crop_roi(img, ROI)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        b = blur_score(gray)
        g = prep_for_diff(gray)

        if prev is None:
            # frame đầu tiên luôn giữ để có baseline
            d = 1.0
            ok = True
        else:
            d = diff_ratio(prev, g)
            ok = (b >= BLUR_THR) and (d >= DIFF_THR)

        r["do_ocr"] = bool(ok)
        r["gating_scores"] = {"blur": b, "diff_ratio": d}

        if ok:
            kept += 1
            prev = g  # update baseline chỉ khi giữ (ổn định hơn)
        # nếu bỏ, không update prev để tránh “drift” do blur/motion

    write_jsonl(OUT_PATH, rows)
    print(f"Total: {len(rows)}, Kept for OCR: {kept} ({kept/len(rows)*100:.1f}%)")
    print(f"Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()