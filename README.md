# RefHCM: A Unified Model for Referring Perceptions in Human-Centric Scenarios

<h5 align="left">
    
[![hugging_face](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/JJJYmmm/RefHCM)
[![arXiv](https://img.shields.io/badge/Arxiv-2412.14643-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.14643) <br>

</h5>

## Overview

Here is the official implementation of **RefHCM**, a unified model designed specifically for human-centric scenarios, enabling it to perform several referring perception tasks. 

![Architecture](examples/arch.jpg)

## Capabilities of RefHCM

RefHCM paves the way for advanced referring abilities in human-AI interactions. For current applications, it can simplify the AIGC content generation pipeline. 

Similar to [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2/blob/main/nodes.py), RefHCM provides addtional keypoint information for specified individuals and more fine-grained human part segmentation results, which can be utilized for tasks like dance generation and image editing.  By the way, we are also considering integrating RefHCM into ComfyUI to further expand its utility...

## Todo List
- [x] Add S1-like sampling method to perform multiple tasks in one instruction! See `demo_s1.py`.
- [x] Release the data and model before January 1, 2025
- [x] Release the code before December 15, 2024
- [ ] Integrate RefHCM into ComfyUI

## Requirements

- python 3.7.4
- pytorch 1.8.1
- torchvision 0.9.1

## Installation

```bash
git clone https://github.com/JJJYmmm/RefHCM
pip install -r requirements.txt
```

For environment setup issues, e.g. `fairseq` installation, refer to the manual setup guide in [Google Colab](https://colab.research.google.com/drive/1AHQNRdaUpRTgr3XySHSlba8aXwBAjwPB?usp=sharing). (recommended)

## Quick Start

- Download the model weight refhcm.pt from [here](https://github.com/JJJYmmm/RefHCM/tree/main/checkpoints), and put it in folder `/checkpoints`

- Launch the gradio demo

  ```bash
  CUDA_VISIBLE_DEVICES=0 python gradio_demo.py
  ```

- Now you can try RefHCM 😊, here are some examples.

  <img src="examples\rhrc.png" alt="rhrc" style="zoom:20%;" />

  <img src="examples\rkpt.png" alt="rkpt" style="zoom: 20%;" />

  <img src="examples\rpar.png" alt="rpar" style="zoom:33%;" />

## Data Preparation and Pretrained Model

Please refer to [RefHCM/checkpoints at main · JJJYmmm/RefHCM](https://github.com/JJJYmmm/RefHCM/tree/main/checkpoints) and [RefHCM/dataset at main · JJJYmmm/RefHCM](https://github.com/JJJYmmm/RefHCM/tree/main/dataset)

## Training and Evaluate

We provide training and evaluate scripts in `/run_scripts` folder, including single-task and multi-task training.

> The scripts are designed to be plug-and-play, assuming you have followed the data preparation and pretrained model setup instructions.

### Referring Expression Comprehension (REC)

```bash
cd run_script/rec/
bash train_refcoco.sh # training
bash evaluate_refcoco.sh # evaluate
```

### Referring Keypoint (RKpt)

```bash
cd run_script/rkpt/
bash train_rkpt.sh # training
bash evaluate_rkpt.sh # evaluate
```

### Referring Parsing (RPar)

`full_mask` means Query Parallel Generation (QPG) mentioned in the paper, which can speed up the generation speed while retains most of the performance.

```bash
cd run_script/rpar/
bash train_rpar.sh # training
bash evaluate_rpar.sh # evaluate

bash train_rpar_full_mask.sh # training for QPG
bash evaluate_rpar_full_mask.sh # evaluate for QPG
```

### Referring Human-Related Caption (RHrc)

```bash
cd run_script/rhrc/
bash train_rhrc.sh # training
bash evaluate_rhrc.sh # evaluate
```

### Multi-task Training

```bash
cd run_script/multitask/
bash train_multitask.sh # training, including multitask learning \
		        # and reasoning ablity boosting (RefHCM-tuned)
```

## Results

### Referring Expression Comprehension


|     Model     | Size | Refcoco testA | Refcoco+ testA |
| :-----------: | :--: | :-----------: | :------------: |
|     PFOS      |  -   |     81.94     |     72.43      |
|    UNITER     | 870M |     87.04     |     81.45      |
| OFA-L-refcoco | 520M |     92.93     |     89.87      |
|   UNINEXT-H   |  1B  |     94.33     |     89.63      |
|    RefHCM     | 500M |   **93.69**   |     89.56      |

### Referring Keypoint

The performance results on the Refpose/+/g datasets, introduced in this paper, are presented below:

| Model            | Size | Refpose             | Refpose+            | Refposeg        |
| ---------------- | ---- | ------------------- | ------------------- | --------------- |
| Unified-IO-2     | 1.1B | 89.13/52.00         | 82.35/48.25         | 89.94/54.01     |
| $PoseGPT_{text}$ | 500M | 78.70/70.50         | 82.03/71.46         | 91.94/**76.77** |
| RefHCM           | 500M | **93.69**/**75.60** | **89.56**/**72.24** | **93.42**/75.69 |

For the RefHuman dataset(text branch), which was proposed in the [RefHuman](https://github.com/bo-miao/RefHuman), the zero-shot performance of RefHCM is listed below.

| Model  | Size | OKS AP   |
| ------ | ---- | -------- |
| UniPHD | 184M | 66.7     |
| RefHCM | 500M | **66.8** |

### Referring Parsing (RPar)

| Model        | Size | mIoU      |
| ------------ | ---- | --------- |
| Florence-2   | 770M | 6.29      |
| Unified-IO-2 | 1.1B | 6.83      |
| RefHCM       | 500M | **45.62** |

### Referring Human-Related Caption (RHrc)



| Model        | Size | CIDEr     |
| ------------ | ---- | --------- |
| Florence-2   | 770M | 0.11      |
| Unified-IO-2 | 1.1B | 0.98      |
| LLaVA-v1.5   | 7B   | 9.54      |
| RefHCM       | 500M | **82.41** |

## Acknowledgments

- [OFA](https://github.com/OFA-Sys/OFA) for their contribution with the training framework.
- [UniHCP](https://github.com/OpenGVLab/UniHCP) for providing metric calculations, such as mIoU.

## Cite

If you find this repository useful, please consider citing it:

```
@misc{refhcm24,
      title={RefHCM: A Unified Model for Referring Perceptions in Human-Centric Scenarios}, 
      author={Jie Huang and Ruibing Hou and Jiahe Zhao and Hong Chang and Shiguang Shan},
      year={2024},
      eprint={2412.14643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.14643}, 
}
```

