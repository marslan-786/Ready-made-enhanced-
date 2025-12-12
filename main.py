from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image
import numpy as np
import cv2
from gfpgan import GFPGANer
import torch
import gc

app = FastAPI()

# Global Variable
restorer = None

# --------------------------------------------
# LAZY LOAD: GFPGAN (The Open Source Remini)
# --------------------------------------------
def get_restorer():
    global restorer
    if restorer is None:
        print("💎 Loading GFPGAN (Remini Model)...")
        # yeh line khud hi model download kr le gi
        restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=2,          # Picture ko 2 guna barra karega
            arch='clean',       
            channel_multiplier=2,
            bg_upsampler=None   # Background ke liye (Optional, RAM bachane ke liye None rakha hai)
        )
    return restorer

# --------------------------------------------
# MAIN API
# --------------------------------------------
@app.get("/enhance")
def enhance(url: str):
    
    # 1. Download Image
    try:
        content = requests.get(url, timeout=10).content
        # PIL Image load karen
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        # GFPGAN ko CV2 format (Numpy array) chahiye hota hai
        cv2_img = np.array(pil_img)
        # RGB to BGR (CV2 format) convert
        cv2_img = cv2_img[:, :, ::-1]
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

    print("⚡ Enhancing Face like Remini...")

    try:
        # 2. Get Model
        model = get_restorer()

        # 3. Magic Happens Here (Remini Logic)
        # yeh function 3 cheezen wapas karta hai, hamein sirf 'restored_img' chahiye
        _, _, restored_img = model.enhance(cv2_img, has_aligned=False, only_center_face=False, paste_back=True)

        # 4. Convert back to PIL to send to user
        # BGR to RGB
        restored_img = restored_img[:, :, ::-1]
        output_pil = Image.fromarray(restored_img)

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    # 5. Return Image
    buf = io.BytesIO()
    output_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    gc.collect()

    return StreamingResponse(buf, media_type="image/jpeg")

@app.get("/")
def home():
    return {"status": "Remini Clone Active"}
  
