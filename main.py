from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import cv2
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

def deskew_image_strict(pil_img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    closed = cv2.morphologyEx(closed := binary.copy(), cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 2, maxLineGap=20)

    if lines is None:
        print("[⚠️] No lines found. Skipping deskew.")
        return pil_img

    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
              for [[x1, y1, x2, y2]] in lines if -45 < np.degrees(np.arctan2(y2 - y1, x2 - x1)) < 45]

    if not angles:
        print("[⚠️] No valid angles detected.")
        return pil_img

    median_angle = np.median(angles)
    print(f"[🧭] Strict deskew angle: {median_angle:.2f}°")

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    return Image.fromarray(rotated).convert("RGB")

def crop_to_text(pil_img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        print("[❌] No content to crop.")
        return pil_img
    x, y, w, h = cv2.boundingRect(coords)
    cropped = pil_img.crop((x, y, x + w, y + h))
    print(f"[✂️] Cropped area: x={x}, y={y}, w={w}, h={h}")
    return cropped

def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)
    contrast = ImageEnhance.Contrast(gray).enhance(1.2)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.5)
    return sharpened.convert("RGB")

@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = ImageOps.exif_transpose(image)
        aligned = deskew_image_strict(image)
        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=aligned_image.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/enhance-ocr")
async def enhance_ocr(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = ImageOps.exif_transpose(image)
        aligned = deskew_image_strict(image)
        cropped = crop_to_text(aligned)
        enhanced = enhance_image(cropped)
        img_bytes = BytesIO()
        enhanced.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/png", headers={
            "Content-Disposition": "inline; filename=enhanced_output.png"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = ImageOps.exif_transpose(image)
        aligned = deskew_image_strict(image)
        cropped = crop_to_text(aligned)
        enhanced = enhance_image(cropped)
        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return {"text": text.strip()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
