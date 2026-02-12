import json
from pathlib import Path
from typing import Any, Dict, List
import cv2

from paddleocr import PaddleOCR

# =======================
# CONFIG
# =======================

IN_PATH = Path("frames_gated.jsonl")
OUT_PATH = Path("frames_ocr.jsonl")

# Resize scale để tăng chất lượng OCR (board video rất cần)
RESIZE_SCALE = 2.0

# =======================
# INIT OCR (chỉ 1 lần)
# =======================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# =======================
# UTILS
# =======================

def robust_collect_text(result: Any) -> List[str]:
    """
    Collect all text recursively from PaddleOCR output
    Works for both old & new pipeline formats
    """
    lines: List[str] = []

    def visit(x):
        if x is None:
            return

        if isinstance(x, str):
            s = x.strip()
            if s:
                lines.append(s)
            return

        if isinstance(x, dict):
            for k in ("text", "rec_text", "transcription"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    lines.append(v.strip())

            for v in x.values():
                visit(v)
            return

        if isinstance(x, (list, tuple)):
            for it in x:
                visit(it)
            return

    visit(result)

    # remove duplicates but keep order
    seen = set()
    out = []
    for t in lines:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out


def extract_text(img_path: str) -> str:
    p = Path(img_path)

    if not p.exists():
        print(f"[WARNING] Image not found: {p.resolve()}")
        return ""

    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARNING] OpenCV cannot read image: {p.resolve()}")
        return ""

    # Resize to improve OCR on lecture videos
    if RESIZE_SCALE != 1.0:
        img = cv2.resize(
            img,
            None,
            fx=RESIZE_SCALE,
            fy=RESIZE_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

    result = ocr.predict(img)

    lines = robust_collect_text(result)

    return "\n".join(lines)


# =======================
# MAIN
# =======================

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH.resolve()}")

    total = 0
    gated = 0
    ocred = 0

    BASE_DIR = IN_PATH.resolve().parent

    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            obj: Dict[str, Any] = json.loads(line)
            total += 1

            do_ocr = bool(obj.get("do_ocr", False))

            if do_ocr:
                gated += 1

                if obj.get("ocr") in (None, ""):
                    rel_path = obj.get("path")
                    if not rel_path:
                        print(f"[WARNING] Missing path at line {line_no}")
                        continue

                    img_path = (BASE_DIR / rel_path).resolve()

                    print(f"OCR frame_id={obj.get('frame_id')} "
                          f"t={obj.get('timestamp_s')} "
                          f"path={img_path}")

                    text = extract_text(str(img_path))
                    obj["ocr"] = text if text else None
                    ocred += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("=================================")
    print(f"Total rows: {total}")
    print(f"Gated (do_ocr=True): {gated}")
    print(f"OCR performed: {ocred}")
    print(f"Output file: {OUT_PATH.resolve()}")
    print("=================================")


if __name__ == "__main__":
    main()