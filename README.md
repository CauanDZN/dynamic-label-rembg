# 🚀 Dynamic Label RemBG - Google Colab

> Remove background from images and generate YOLO labels automatically.

---

## Quick Start

1. **Upload images** to Colab folder (or connect Google Drive)
2. **Run cells** below in order

---

## Setup

```python
# Install dependencies
!pip install numpy Pillow "rembg[gpu]" -q

# Import and config
from pathlib import Path
from PIL import Image, ImageDraw, ImageFile
from rembg import remove, new_session
import numpy as np
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | `.` | Folder with input images |
| `OUTPUT_DIR` | `./output` | Folder for processed output |
| `IMAGE_MAX_SIZE` | `640` | Max image dimension (px) |
| `CLASS_ID` | `0` | YOLO class ID |
| `WORKERS` | `4` | Parallel workers |
| `ALPHA_THRESHOLD` | `10` | Alpha transparency cutoff (higher = stricter) |
| `MIN_OBJECT_SIZE` | `100` | Min pixels to consider valid object |
| `MODEL_NAME` | `isnet-general-use` | Model: u2net, u2netp, silueta, isnet-general-use, birefnet-general |

---

## Processing

### 1. Session Setup

```python
import torch
providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else []
session = new_session(providers=providers)
```

### 2. Functions

```python
def get_yolo_bbox(img_pil):
    output = remove(img_pil, session=session)
    alpha = np.array(output)[:, :, 3]
    pos = np.where(alpha > ALPHA_THRESHOLD)
    
    if pos[0].size == 0:
        return None

    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    
    w, h = img_pil.size
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    cx = (xmin + (xmax - xmin) / 2) / w
    cy = (ymin + (ymax - ymin) / 2) / h
    
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
```

### 3. Process All Images

```python
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("./output")
DEBUG_DIR = OUTPUT_DIR / "debug_visual"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def process_file(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if max(img.size) > IMAGE_MAX_SIZE:
                img.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
            
            yolo_label = get_yolo_bbox(img)
            
            if yolo_label:
                name = img_path.stem
                img.save(OUTPUT_DIR / "images" / f"{name}.jpg", "JPEG", quality=90)
                (OUTPUT_DIR / "labels" / f"{name}.txt").write_text(yolo_label)
                
                # Debug visualization
                debug_img = img.copy()
                # Draw bbox here if needed
                debug_img.save(DEBUG_DIR / f"check_{name}.jpg")
                return True
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
print(f"Processing {len(files)} images...")

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    results = list(executor.map(process_file, files))

successes = sum(1 for r in results if r)
print(f"Done! {successes}/{len(files)}")
```

---

## Output Structure

```
output/
├── images/          # Processed JPG images
├── labels/         # YOLO format txt files
└── debug_visual/  # Debug images with bbox
```

---

## Tips

- **GPU**: Colab provides free GPU. `rembg[gpu]` uses CUDA automatically.
- **Batch size**: Adjust `WORKERS` based on GPU memory.
- **Alpha threshold**: Lower = more aggressive removal. Higher = only solid pixels.
- **Multiple classes**: Run script multiple times with different `CLASS_ID`.