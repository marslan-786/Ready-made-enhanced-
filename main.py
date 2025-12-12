from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests
import io
import numpy as np
import cv2
import torch
import gc
from facexlib.utils.face_restoration_helper import FaceRestorationHelper
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# Global Helper
face_helper = None
bg_upsampler = None

def load_models():
    global face_helper, bg_upsampler
    
    if face_helper is None:
        print("🚀 Loading CodeFormer (The Beast)...")
        
        # 1. Background Enhancer (RealESRGAN)
        # Yeh kapron aur background ko sharp karega
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False 
        )

        # 2. CodeFormer Setup
        # 'CodeFormer' select kia hai instead of GFPGAN
        face_helper = FaceRestorationHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='jpg',
            use_parse=True,
            device=torch.device('cpu')
        )
        print("✅ CodeFormer Loaded successfully!")

    return face_helper, bg_upsampler

@app.get("/enhance")
def enhance(url: str, fidelity: float = 0.6):
    # fidelity: 0.0 (High Quality, AI Face) -> 1.0 (Low Quality, Original Face)
    # 0.6 is best balance for "VIP" look.
    
    print(f"⚡ Processing with CodeFormer: {url}")
    
    try:
        # 1. Download
        resp = requests.get(url, stream=True, timeout=20)
        image_arr = np.asarray(bytearray(resp.content), dtype="uint8")
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid Image"}

        # 2. Load Models
        helper, bg_model = load_models()

        # 3. Clean previous data
        helper.clean_all()

        # 4. Detect Faces & Align
        # Pehle background ko upscale karo
        bg_img = bg_model.enhance(img, outscale=2)[0]
        
        # Ab faces dhundo
        helper.read_image(bg_img)
        # Face detection run karo
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        helper.align_warp_face()

        # 5. Run CodeFormer Inference
        # Yeh line asal magic hai (CodeFormer restoration)
        print("Running CodeFormer restoration...")
        helper.restore_face(model_name='CodeFormer', fidelity=fidelity)

        # 6. Paste faces back to background
        helper.get_inverse_affine(None)
        final_img = helper.paste_faces_to_input_image()

        # 7. Convert to JPEG
        _, encoded_img = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        gc.collect()

        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.get("/")
def home():
    return {"status": "CodeFormer Active", "info": "The Best Face Restoration Model"}
