from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import base64

app = FastAPI()

# ---------- Deskew ----------
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
    print(f"[🧭] Deskew angle: {median_angle:.2f}°")

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    return Image.fromarray(rotated).convert("RGB")

# ---------- Enhance ----------
def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.Resampling.LANCZOS)
    contrast = ImageEnhance.Contrast(gray).enhance(1.2)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.5)
    return sharpened.convert("RGB")

# ---------- Auto-crop ----------
def autocrop(pil_img: Image.Image) -> Image.Image:
    img_array = np.array(pil_img)
    if img_array.ndim == 2:
        mask = img_array < 250
    else:
        mask = np.mean(img_array, axis=2) < 250
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return pil_img.crop((x0, y0, x1, y1))
    return pil_img

# ---------- JPEG Output ----------
def prepare_outputs(image: Image.Image, max_width=1000, quality=85):
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    img_bytes = buffer.getvalue()

    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    base64_url = f"data:image/jpeg;base64,{base64_image}"

    return {
        "image_base64_url": base64_url,
        "image_binary": base64_image,
        "mime_type": "image/jpeg",
        "file_name": "enhanced_image.jpg"
    }

# ---------- PDF Output ----------
def prepare_pdf_output(image: Image.Image, pdf_filename="enhanced_output.pdf"):
    pdf_buffer = BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(pdf_buffer, format="PDF")
    pdf_bytes = pdf_buffer.getvalue()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    base64_pdf_url = f"data:application/pdf;base64,{base64_pdf}"

    return {
        "pdf_base64_url": base64_pdf_url,
        "pdf_binary": base64_pdf,
        "mime_type": "application/pdf",
        "file_name": pdf_filename
    }

# ---------- Main Endpoint ----------
@app.post("/enhance-image")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        enhanced = enhance_image(aligned)
        cropped = autocrop(enhanced)

        image_result = prepare_outputs(cropped)
        pdf_result = prepare_pdf_output(cropped)

        return JSONResponse(content={
            "image_result": image_result,
            "pdf_result": pdf_result
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
