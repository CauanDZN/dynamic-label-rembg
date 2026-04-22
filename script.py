import os
import warnings
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFile, ImageDraw
from rembg import remove, new_session
from concurrent.futures import ThreadPoolExecutor

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("./output")
DEBUG_DIR = OUTPUT_DIR / "debug_visual"
IMAGE_MAX_SIZE = 640
CLASS_ID = 0
WORKERS = 4
ALPHA_THRESHOLD = 10  # Higher = more strict removal of semi-transparent pixels
MIN_OBJECT_SIZE = 100  # Min pixels to consider as valid object
MODEL_NAME = "isnet-general-use"  # Higher quality than u2net

providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else []
session = new_session(model_name=MODEL_NAME, providers=providers)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_yolo_bbox_and_draw(img_pil):
    output = remove(img_pil, session=session)
    alpha = np.array(output)[:, :, 3]
    
    pos = np.where(alpha > ALPHA_THRESHOLD)
    
    if pos[0].size == 0:
        return None, None, None

    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    
    w_img, h_img = img_pil.size
    
    width = (xmax - xmin) / w_img
    height = (ymax - ymin) / h_img
    x_center = (xmin + (xmax - xmin) / 2) / w_img
    y_center = (ymin + (ymax - ymin) / 2) / h_img
    
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    if width <= 0 or height <= 0:
        return None, None, None
    
    pixel_count = (xmax - xmin) * (ymax - ymin)
    if pixel_count < MIN_OBJECT_SIZE:
        return None, None, None
    
    yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    debug_img = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    
    return yolo_line, debug_img, output

def process_file(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if max(img.size) > IMAGE_MAX_SIZE:
                ratio = IMAGE_MAX_SIZE / max(img.size)
                new_w = max(1, int(img.width * ratio))
                new_h = max(1, int(img.height * ratio))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            yolo_label, debug_img, _ = get_yolo_bbox_and_draw(img)
            
            if yolo_label:
                name = img_path.stem
                img.save(OUTPUT_DIR / "images" / f"{name}.jpg", "JPEG", quality=90)
                (OUTPUT_DIR / "labels" / f"{name}.txt").write_text(yolo_label)
                debug_img.save(DEBUG_DIR / f"check_{name}.jpg")
                return True
            return False
    except Exception as e:
        print(f"Error in {img_path.name}: {e}")
        return False

def main():
    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    print(f"Processing {len(files)} images with bbox visualization...")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(process_file, files))

    successes = sum(1 for r in results if r)
    print(f"\nDone!")
    print(f"Success: {successes}/{len(files)}")
    print(f"Check visualizations: {DEBUG_DIR.absolute()}")

if __name__ == "__main__":
    main()