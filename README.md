## Guideline


**1. Create virtual environment & Install dependencies**
```bash
python -m venv .venv
pip install -r requirements.txt
```

**2. Extract audio and keyframes**
```bash
python extract.py
```

**3. OCR text on keyframes**
```bash
python ocr.py
```