import json
from pathlib import Path
import re

KEYFRAMES_DIR = Path("keyframes")
OUT_PATH = Path("frames.jsonl")

# nếu bạn dùng fps=1
FPS = 1.0

pat = re.compile(r"output_(\d+)\.jpg$")

rows = []
for p in sorted(KEYFRAMES_DIR.glob("output_*.jpg")):
    m = pat.search(p.name)
    if not m:
        continue
    # output_00001.jpg -> idx=1
    idx = int(m.group(1))
    frame_id = idx - 1  # để frame_id bắt đầu từ 0
    timestamp_s = frame_id / FPS

    rows.append({
        "frame_id": frame_id,
        "timestamp_s": float(timestamp_s),
        "path": str(p.as_posix()),
        "ocr": None,         # field bạn muốn điền OCR sau này
    })

with OUT_PATH.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows to {OUT_PATH}")