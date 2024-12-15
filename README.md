# RefHCM: A Unified Model for Referring Perceptions in Human-Centric Scenarios

## Overview

Here is the official implementation of **RefHCM**, a unified model designed specifically for human-centric scenarios, enabling it to perform several referring perception tasks. 

![Architecture](examples/arch.jpg)

## Capabilities of RefHCM

RefHCM paves the way for advanced referring abilities in human-AI interactions. For current applications, it can simplify the AIGC content generation pipeline. 

Similar to [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2/blob/main/nodes.py), RefHCM provides addtional keypoint information for specified individuals and more fine-grained human part segmentation results, which can be utilized for tasks like dance generation and image editing.  By the way, we are also considering integrating RefHCM into ComfyUI to further expand its utility...

## Todo List

- [x] Release the code before December 15, 2024
- [ ] Release the data and model before January 1, 2025
- [ ] Integrate RefHCM into ComfyUI (no specific deadline)

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

- Download the model weight [here](xxx), and put it in folder `/checkpoints`

- Launch the gradio demo

  ```bash
  CUDA_VISIBLE_DEVICES=0 python gradio_demo.py
  ```

- Now you can try RefHCM:smile:, here are some examples.

  <img src="C:\Users\Axuanz\Desktop\refhcm\examples\rec.png" alt="rec" style="zoom:20%;" /><img src="C:\Users\Axuanz\Desktop\refhcm\examples\rkpt.png" alt="rkpt" style="zoom: 20%;" />

  <img src="C:\Users\Axuanz\Desktop\refhcm\examples\rpar.png" alt="rpar" style="zoom:33%;" />

  <img src="C:\Users\Axuanz\Desktop\refhcm\examples\rhrc.png" alt="rhrc" style="zoom:20%;" />

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

`full_mask` means Query Parallel Generation (QPG) mentioned in the paper, which can speed up the generation speed while remains the performance.

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

## Acknowledgments

- [OFA](https://github.com/OFA-Sys/OFA) for their contribution with the training framework.
- [UniHCP](https://github.com/OpenGVLab/UniHCP) for providing metric calculations, such as mIoU.

## Cite

If you find this repository useful, please consider citing it:

```
@misc{refhcm2024,
  author = {Jie Huang},
  title = {{RefHCM: A Unified Model for Referring Perceptions in Human-Centric Scenarios}},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JJJYmmm/RefHCM}},
}
```

