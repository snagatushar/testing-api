from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import base64

app = FastAPI()

# Deskew image using OpenCV
def deskew_image_strict(pil_img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 2, maxLineGap=20)

    if lines is None:
        print("[⚠️] No lines found. Skipping deskew.")
        return pil_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

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

# Pillow enhancement
def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)
    contrast = ImageEnhance.Contrast(gray).enhance(1.2)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.5)
    return sharpened.convert("RGB")

# Convert image to base64
def encode_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}", img_bytes

@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        base64_img, binary_data = encode_base64(aligned)

        return JSONResponse(
            content={
                "filename": "aligned_image.png",
                "content_type": "image/png",
                "base64": base64_img
            },
            headers={
                "Content-Disposition": "attachment; filename=aligned_image.png"
            },
            media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/enhance-image")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        enhanced = enhance_image(aligned)
        base64_img, binary_data = encode_base64(enhanced)

        return JSONResponse(
            content={
                "filename": "enhanced_image.png",
                "content_type": "image/png",
                "base64": base64_img
            },
            headers={
                "Content-Disposition": "attachment; filename=enhanced_image.png"
            },
            media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
