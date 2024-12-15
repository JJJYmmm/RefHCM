import torch
import numpy as np
from torch.nn import functional as F
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.refcoco import RefcocoTask
from tasks.mm_tasks.multitask import MultiTaskTask
from models.taming.models.vqgan import VQSegmentationModel

from models.refhcm import OFAModel
from PIL import Image

import os
import re
import cv2
import random
import sys
import argparse

sys.path.append('../')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(7)

args = argparse.ArgumentParser()
args.add_argument('--image_path', type=str, required=True)
args.add_argument('--text', type=str, required=True)
args.add_argument('--kpt', default=False, action='store_true')
args.add_argument('--par', default=False, action='store_true')
flags = args.parse_args()

tasks.register_task('refcoco', RefcocoTask)
tasks.register_task('mutitask', MultiTaskTask)

use_cuda = torch.cuda.is_available()
use_fp16 = True

parser = options.get_generation_parser()
input_args = ["", "--task=multitask", "--beam=5", "--path=checkpoints/ofa_multitask.pt", "--bpe-dir=utils/BPE", "--patch-image-size=512"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# load model
task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths('checkpoints/ofa_multitask.pt'),
    task=task
)
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# load vq
from omegaconf import OmegaConf
vq_n_embed = 48
vq_config="checkpoints/vqgan/model.yaml"
vq_ckpt="checkpoints/vqgan/model.ckpt"
config = OmegaConf.load(vq_config)
vqgan = VQSegmentationModel(**config.model.params)
sd = torch.load(vq_ckpt, map_location="cpu")["state_dict"]
missing, unexpected = vqgan.load_state_dict(sd, strict=False)
for k, v in vqgan.named_parameters():
    v.requires_grad = False
vqgan.cuda()
vqgan.eval()

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 200
# cfg.generation.no_repeat_ngram_size = 3
# task.cfg.constraint_range = "58457,59457"
generator = task.build_generator(models, cfg.generation)

from torchvision import transforms
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    total_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
        total_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
        total_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
          total_result.append(token.strip())
        else:
          token_result[-1] += token
          total_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result), ' '.join(total_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip())) 
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

def extract_number(string):
    pattern = r'<bin_(\d+)>'
    match = re.search(pattern, string)
    if match:
        return int(match.group(1))
    else:
        return None

kpt_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
kpt2idx = {item: idx for idx, item in enumerate(kpt_name)}
sk = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

def process_kpt(tokens, bins, w_ratio, h_ratio):
    tokens = tokens.strip().split()
    kpts = bins.strip().split()[4:] # jump bbox
    length_token = len(tokens)
    length_kpts = len(kpts)
    if length_token * 2 > length_kpts:
        length_token = length_kpts // 2

    result = np.zeros((17,2))
    for i in range(length_token):
        pos = kpt2idx.get(tokens[i])
        if pos == None:
            continue
        result[pos][0] = extract_number(kpts[2*i])
        result[pos][1] = extract_number(kpts[2*i+1])
    
    result = result / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    result[:, ::2] /= w_ratio
    result[:, 1::2] /= h_ratio

    print(result)
    return result.tolist()

def vis_kpt(kpts, img):

    for kpt in kpts:
        x, y = kpt
        cv2.circle(img,(int(x),int(y)), 4, (255, 255, 0), -1)
    
    for i in range(len(sk)):
        [a,b]=sk[i]
        if kpts[a-1] != [0,0] and kpts[b-1]!=[0,0]:
            cv2.line(img,(int(kpts[a-1][0]),int(kpts[a-1][1])),(int(kpts[b-1][0]),int(kpts[b-1][1])), (0, 255, 0), 2)
    
    cv2.imwrite('./kpt_result.jpg', img)
    
                        

def process_img(codes, coord_list, img_w, img_h):
    codes = [int(code[6:-1]) for code in codes.strip().split()]
    assert len(codes) == 48 # hard code for 8*6 latent codes
    codes = torch.tensor(codes).unsqueeze(0)
    B = codes.shape[0]
    codes = codes.clamp(0,vq_n_embed - 1).to(vqgan.device)
    codes = F.one_hot(codes, num_classes = vqgan.quantize.embedding.weight.size(0))
    quant  = codes.to(vqgan.dtype) @ vqgan.quantize.embedding.weight
    quant = quant.reshape(B, 8, 6, 256).permute(0,3,1,2) # TODO: hard code for 8*6 latent code
    output = vqgan.decode(quant)
    output = torch.argmax(output, dim=1)

    pred = torch.zeros(img_h, img_w).to(output.device)
    x1, y1, x2, y2 = map(int,coord_list)

    pred[y1:y2, x1:x2] = F.interpolate(output[0].to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=(y2 - y1, x2 - x1), mode='nearest')
    pred = pred.to(dtype=torch.int32)
    return pred
   
def vis_par(x, img, n_labels = 20):
    color = torch.randn(3, n_labels, 1, 1).to(x.device)
    x = F.one_hot(x.unsqueeze(0).to(torch.long), num_classes=n_labels).permute(0,3,1,2).float()
    x = F.conv2d(x, weight=color)
    x = (x-x.min())/(x.max()-x.min())
    mask = transforms.ToPILImage()(x.squeeze(0))
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    result = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    cv2.imwrite('./par_result.jpg', result)

if __name__ == '__main__':
    image_file = os.path.join(flags.image_path)
    image = Image.open(image_file)
    w, h = image.size
    w_resize_ratio = task.cfg.patch_image_size / w
    h_resize_ratio = task.cfg.patch_image_size / h

    prompt = f'which region does the text "{flags.text}" describe?'
    if flags.kpt:
        prompt += ' Provide the bounding box and keypoints.'
    if flags.par:
        prompt += ' Provide the bounding box and the parsing map.'

    print(prompt)

    sample = construct_sample(image, prompt)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        tokens, bins, imgs, result = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        print(f'tokens:{tokens}')
        print(f'bins:{bins}')
        print(f'imgs:{imgs}')
        print(f'result:{result}')

        coord_list = bin2coord(bins, w_resize_ratio, h_resize_ratio)

        print(f'region: {coord_list[:4]}')

        # location
        image_vis = cv2.imread(image_file)
        cv2.rectangle(
            image_vis,
            (int(coord_list[0]), int(coord_list[1])),
            (int(coord_list[2]), int(coord_list[3])),
            (0, 255, 0),
            3
        )
        cv2.imwrite('./loc_result.jpg', image_vis)

        # kpt
        if flags.kpt:
            kpts = process_kpt(tokens, bins, w_resize_ratio, h_resize_ratio)
            vis_kpt(kpts, image_vis)

        # par
        if flags.par:
            par_img = process_img(imgs, coord_list, w, h)
            vis_par(par_img, image_vis)
