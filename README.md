# DecodeLabs AI Project 4 - Image / Text Recognition (Basic)

Pre-trained, 100% free stack: **OpenCV + Tesseract (pytesseract) + YOLOv4-tiny (COCO)**. No API keys.

## Setup
```bash
# Tesseract engine: Windows -> https://github.com/UB-Mannheim/tesseract/wiki | macOS -> brew install tesseract | Linux -> sudo apt install tesseract-ocr
pip install -r requirements.txt
```

## Run
```bash
python recognition.py ocr    samples/sample_text.png          # Path 1: OCR
python recognition.py detect your_photo.jpg                   # Path 2: object detection (model auto-downloads once, ~20 MB)
python recognition.py ocr    invoice.png --psm 11             # tune page segmentation
python server.py                                              # web UI -> http://127.0.0.1:5000
```
Outputs land in `outputs/` (annotated image, JSON report, and pre-processing step images).

## How it maps to the kit's 4 validation gates
| Gate | Where |
|---|---|
| 1. Library integration | `pytesseract.image_to_data`, `cv2.dnn_DetectionModel` |
| 2. Pre-processing integrity | `preprocess()`: grayscale -> Gaussian blur -> deskew -> adaptive threshold (step images saved) |
| 3. >= 80 % confidence | every word/box is gated by `CONF_THRESHOLD = 0.80`; mean confidence reported |
| 4. Visual confirmation | boxes + confidence drawn on the image, saved to `outputs/` |

## Sample result (included)
Skewed + noisy test image: deskew -3.95 deg, 17 words, **mean confidence 94.1 %**, 0 dropped, validation PASS.

## Notes
- Detection uses YOLOv4-tiny v3 (`dnn_DetectionModel`: 320x320 resize, mean subtraction, scale 1/127.5, RB swap, NMS).
- If auto-download is blocked, the script prints the exact URLs to fetch manually into `models/`.

## Test images (samples/) - what to expect
| File | Tests | Expected |
|---|---|---|
| sample_text.png | tilt + noise | PASS |
| 2_wavy_banner.png | warped text | PASS |
| 3_low_light_shadow.png | dim, uneven light | PASS |
| 4_blur_noise.png | blur + salt-and-pepper | PASS |
| 6_rotated_inverted_sign.png | white text on dark, rotated | PASS |
| 1_curved_arc.png | text on a circular arc | FAIL (Tesseract limit) |
| 5_handwriting_cursive.png | cursive handwriting | FAIL (Tesseract limit) |

The gate now needs mean confidence >= 80% AND >= 60% of detected words above 80%, so one lucky word can't pass an image.
For detection, test with real photos (person, car, dog, laptop); synthetic drawings are not recognised.

## Advanced engine: Gemini (free tier)
1. Get a free key at https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`
3. `python server.py` -> choose **Gemini · any object** in the UI.
Detects any object (not just 80 COCO classes), several per image; OCR reads handwriting, curved and multi-language text.
Confidence is Gemini's own 0-100 estimate (not a calibrated probability). Images are sent to Google's API. Free tier has rate limits.
The **Classic · offline** engine (Tesseract + YOLOv4-tiny) stays available.
