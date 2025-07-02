from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
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

# Convert PIL Image to Base64 URL
def pil_to_base64_url(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.post("/align-image")
async def align_image(file: UploadFile = File(...)):
    """
    Endpoint for deskewing the uploaded image.
    Returns the resulting image in binary format.
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        aligned = deskew_image_strict(image)
        
        # Return the aligned image as binary stream without Base64 conversion
        img_bytes = BytesIO()
        aligned.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return StreamingResponse(
            img_bytes, 
            media_type="image/png", 
            headers={"Content-Disposition": "inline; filename=aligned_image.png"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/enhance-image")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for processing the image by first deskewing it then enhancing it.
    The resulting image is converted to Base64 and returned as a JSON.
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        
        # First, deskew the image
        aligned = deskew_image_strict(image)
        # Then, perform Pillow-based enhancement
        enhanced = enhance_image(aligned)
        
        # Convert enhanced image to Base64 URL
        base64_url = pil_to_base64_url(enhanced)
        return JSONResponse(content={"image_base64_url": base64_url})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
