from pathlib import Path
from paddleocr import PaddleOCR

DATA_PATH = Path("./keyframes")
OUTPUT_PATH = Path("./output")
OUTPUT_PATH.mkdir(exist_ok=True)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

for img in DATA_PATH.glob("*.jpg"):
    print(f"OCR for image: {img.name}")

    result = ocr.predict(str(img))

    for res in result:
        res.print()
        res.save_to_img(str(OUTPUT_PATH))
        res.save_to_json(str(OUTPUT_PATH))
