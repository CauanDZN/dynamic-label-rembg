import os
import warnings
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile, ImageDraw
from rembg import remove, new_session
from concurrent.futures import ThreadPoolExecutor

# Configurações
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("./output")
DEBUG_DIR = OUTPUT_DIR / "debug_visual"  # Nova pasta para ver as marcações
IMAGE_MAX_SIZE = 640
CLASS_ID = 0
WORKERS = 4

session = new_session()

# Criar pastas
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_yolo_bbox_and_draw(img_pil):
    """
    Remove fundo, calcula Bbox e desenha para conferência.
    """
    output = remove(img_pil, session=session)
    alpha = np.array(output)[:, :, 3]
    
    # Sensibilidade: pixels com mais de 10 de opacidade
    pos = np.where(alpha > 10)
    
    if pos[0].size == 0:
        return None, None, None

    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    
    w_img, h_img = img_pil.size
    
    # YOLO format
    bw = (xmax - xmin) / w_img
    bh = (ymax - ymin) / h_img
    cx = (xmin + (xmax - xmin) / 2) / w_img
    cy = (ymin + (ymax - ymin) / 2) / h_img
    yolo_line = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    # --- Criar imagem de Debug ---
    # Fazemos uma cópia da imagem original e desenhamos o retângulo
    debug_img = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    # Desenha o retângulo (xmin, ymin, xmax, ymax)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    
    return yolo_line, debug_img, output

def process_file(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if max(img.size) > IMAGE_MAX_SIZE:
                img.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
            
            yolo_label, debug_img, _ = get_yolo_bbox_and_draw(img)
            
            if yolo_label:
                name = img_path.stem
                # Salva o arquivo de treino
                img.save(OUTPUT_DIR / "images" / f"{name}.jpg", "JPEG", quality=90)
                (OUTPUT_DIR / "labels" / f"{name}.txt").write_text(yolo_label)
                
                # Salva a imagem de visualização para você conferir
                debug_img.save(DEBUG_DIR / f"check_{name}.jpg")
                return True
            return False
    except Exception as e:
        print(f"❌ Erro em {img_path.name}: {e}")
        return False

def main():
    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    print(f"🚀 Verificando {len(files)} imagens com visualização de Bbox...")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(process_file, files))

    successes = sum(1 for r in results if r)
    print(f"\n✅ Concluído!")
    print(f"📦 Sucesso: {successes}")
    print(f"👁️  CONFIRA AS MARCAÇÕES EM: {DEBUG_DIR.absolute()}")

if __name__ == "__main__":
    main()