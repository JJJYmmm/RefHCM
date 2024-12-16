# RefHCM-dataset

Huggingface: https://huggingface.co/datasets/JJJYmmm/RefHCM-dataset

RefHCM-dataset can be categorized into two parts.

- **multi-task training data** in `rec`, `rkpt`, `rpar`, `rhrc` folder
- ***ReasonRef* Benchmark** and corresponding fine-tuning data in `reasonref` folder

## Statics

Download Size: 28.9G; Total Size: 55.5G

For *RPar* task, you should unzip the annos.zip first.

## Data Organization

All data are organized in TSV files.

- **Referring Expression Comprehension (REC)**

  ```
  anno_id \t image_id	\t reference \t bounding box(top-left and bottum-right) \t base64_of_image
  182_1	579997	Blurry person to right of player's elbow	425.68,219.3,542.79,390.11	/9j/xxx
  ```

- **Referring Keypoint (RKpt)**

  ```
  anno_id \t image_id	\t reference \t bounding box(top-left and bottum-right) \t keypoints(17x2) \t base64_of_image
  756_0	573476	left person	112.21,46.13,341.61,266.81	212,110,219,102,196,100,0,0,156,113,215,168,135,170,223,240,161,215,229,248,251,232,188,302,135,306,0,0,0,0,0,0,0,0	/9j/xxx
  ```

  [0, 0] in keypoints means invisible, which would be ignored in training.

- **Referring Parsing (RPar)**

  ```
  anno_id \t caption \t bounding box(top-left and bottum-right) \t parsing code \t path_of_mask \t path_of_image
  0000006_0	a man dressed in a white uniform. He is wearing a hat and a helmet . His attire suggests he might be part of a military or formal ceremony.	112,170,196,459	15,31,26,12,6,27,22,18,5,28,3,15,13,11,9,22,0,26,31,2,21,30,23,28,18,24,10,5,24,5,15,30,29,2,4,2,25,29,14,7,23,23,14,8,12,6,27,23	train/0000006_0.npy	train/0000006.jpg
  ```

  You should specific the root path of annos in [here](https://github.com/JJJYmmm/RefHCM/blob/8619e06dbe57721f632b652e28f2fc720a5fc7c1/data/mm_data/rpar_dataset.py#L33) before training on the *RPar* task or multi-task training.

- **Referring Human-Related Caption (RHrc)**

  ```
  anno_id \t captions \t bounding box(top-left and bottum-right) \t base64_of_image
  0010548_0	The individual is a man with short blonde hair, wearing a blue shirt and grey pants. He is also wearing a black watch on his left wrist.	225,19,364,380	/9j/4AAQS
  ```

*You can recover the original image from its base64 encoding form or get the base64 form by:

```python
from io import BytesIO
from PIL import Image
import base64

def get_base64_from_image(image_path):
    img = Image.open(image_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_str = base64_str.decode("utf-8")
    return base64_str

def get_image_from_base64(base64_str):
	return Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
```

## File Tree

```
dataset
  ├─reasonref
  │  ├─reasondec
  │  ├─reasonpar
  │  │  └─annos
  │  └─reasonpose
  ├─rec
  ├─rhrc
  ├─rkpt
  └─rpar
      └─annos
```
