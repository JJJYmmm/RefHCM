# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math
import json
from itertools import chain
import os

import torch
import torch.distributed as dist
from torchvision import utils as vutils
from fairseq import utils

from data import data_utils
import editdistance


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]

def _calculate_error_rate(hyps, refs):
    """each line is "<text> (None-<index>)" """
    assert (len(hyps) == len(refs))
    err_rates = [
        (editdistance.eval(hyp.split(), ref.split()), len(ref.split())) for hyp, ref in zip(hyps, refs)
    ]
    return err_rates


def eval_caption(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
    return results, None

def eval_regioncaption(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hyps, refs = task._inference(generator, sample, models[0])
    scores = task._calculate_cider_scores(hyps, refs)
    # test ground truth
    # import itertools
    # scores = task._calculate_cider_scores(list(itertools.chain.from_iterable(refs)), refs)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        results.append({"image_id": str(sample_id), "region": sample['region_coords'][i].tolist(), "caption": hyps[i].translate(transtab).strip()})
    return results, scores

def eval_caption_cn(task, generator, models, sample, **kwargs):
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(
            hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator
        )
        results.append(
            {
                "image_id": str(sample_id),
                "caption": detok_hypo_str.strip(),
            }
        )
    return results, None


def eval_ocr(task, generator, models, sample, **kwargs):
    gen_out = task.inference_step(generator, models, sample)
    hyps, refs, results = [], [], []
    for i, sample_id in enumerate(sample["id"].tolist()):
        decode_tokens = decode_fn(gen_out[i][0]["tokens"], task.tgt_dict, task.bpe, generator).strip()
        hyps.append(decode_tokens.strip().replace(" ", ""))
        if sample["target"]:
            refs.append(
                decode_fn(
                    utils.strip_pad(sample["target"][i], task.tgt_dict.pad()),
                    task.tgt_dict, task.bpe, generator
                )
                .strip()
                .replace(" ", "")
            )
        results.append(
            {
                "image_id": str(sample_id),
                "ocr": decode_tokens.strip().replace(" ", ""),
            }
        )
    if refs:
        acc = [1.0 if hyp == ref else 0.0 for hyp, ref in zip(hyps, refs)]
    else:
        acc = None

    return results, acc


def eval_vqa_gen(task, generator, models, sample, **kwargs):
    if kwargs['beam_search_vqa_eval']:
        hypos = task.inference_step(generator, models, sample, prefix_tokens=sample['prefix_tokens'])
        results = []
        for i, sample_id in enumerate(sample["id"].tolist()):
            prefix_len = sample['prefix_tokens'][i].ne(1).sum().item()
            detok_hypo_str = decode_fn(hypos[i][0]["tokens"][prefix_len:], task.tgt_dict, task.bpe, generator)
            results.append({"question_id": int(sample_id), "answer": detok_hypo_str.strip()})
        scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in zip(sample['ref_dict'], results)]
        return results, scores

    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    eos_item = torch.tensor([task.src_dict.eos()])
    pad = task.src_dict.pad()
    valid_result = []
    for valid_answers, valid_constraint_masks in zip(task.valid_answers_list, task.valid_constraint_masks_list):
        valid_size = len(valid_answers)
        valid_tgt_items = [
            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_prev_items = [
            torch.cat([torch.tensor(decoder_prompt), valid_answer])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_constraint_mask_items = [
            torch.cat(
                [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask],
                dim=0
            )
            for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
        ]
        valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad).to(device)
        valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad).to(device)
        valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad).to(device)

        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    results = [{"question_id": int(id), "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    return results, scores


def eval_refpose(task, generator, models, sample, **kwargs):

    gen_out = task.inference_step(generator, models, sample)
    hyps = []
    for i in range(len(gen_out)):
        # hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
        hyps.append(task._decode_keypoints(gen_out[i][0]["tokens"][:-1]))
    hyps = torch.stack(hyps, dim=0)
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    # visualize
    # from .vis_utils import vis_pose_result
    # import cv2
    # import numpy as np
    # from PIL import Image
    # for i, sample_id in enumerate(sample["id"].tolist()):
    #     img = Image.open(f'/home/huangjie/projects/OFA/tmp/img/{sample_id}.jpg').convert("RGB")
    #     img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
    #     vis_pose_result(img, hyps[i,4:].reshape(-1,2).cpu().numpy(), 2, f'/home/huangjie/projects/OFA/tmp/result/{sample_id}')

    scores = task._calculate_oks_score(hyps[:,4:], sample['keypoints'].float(), sample['region_coords'].float())
    results = [
        {"uniq_id": sample_id,
         "box": hyps[i].tolist()[:4],
         "keypoints": hyps[i].tolist()[4:],
         "oks_score": scores[i].item()}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    return results, scores.tolist()

def compute_miou(task, _confusion_matrix):
    import numpy as np
    acc = np.full(task.num_labels, np.nan, dtype=np.float64)
    iou = np.full(task.num_labels, np.nan, dtype=np.float64)
    tp = _confusion_matrix.diagonal().astype(np.float64)
    pos_gt = np.sum(_confusion_matrix, axis=0).astype(np.float64)
    pos_pred = np.sum(_confusion_matrix, axis=1).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    pacc = np.sum(tp) / np.sum(pos_gt)
    return macc, miou, pacc


from torch.nn import functional as F
def eval_refparsing(task, generator, models, sample, full_mask = False, **kwargs):

    if full_mask:
        # full mask
        hyps, refs = task._gready_inference(sample, models[0])
    else:
        # no full mask
        gen_out = task.inference_step(generator, models, sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(task._decode_parsing(gen_out[i][0]["tokens"][:-1]))
            refs.append(task._decode_parsing(sample["target"][i][:-1]))
        hyps, refs = torch.stack(hyps, dim=0), torch.stack(refs, dim=0)

    pred_coords, gt_coords = hyps[:,:4], refs[:,:4] # x1y1x2y2
    pred_coords = pred_coords / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    gt_coords = gt_coords / (task.cfg.num_bins - 1) * task.cfg.max_image_size

    if len(task.cfg.vq_config):
        parsing_map = task._get_parsing_map(hyps[:,4:])
    else:
        parsing_map = task._get_parsing_map_titok(hyps[:,4:])

    # align bbox
    N = hyps.size(0)
    pred = torch.zeros(N, task.cfg.max_image_size, task.cfg.max_image_size).to(hyps.device)
    for i in range(N):
        x1, y1, x2, y2 = list(map(lambda x: min(task.cfg.max_image_size,max(0,x)),map(int, pred_coords[i].tolist())))
        x1 = max(0, min(x1, x2-1))
        y1 = max(0, min(y1, y2-1))
        pred[i, y1:y2, x1:x2] = F.interpolate(parsing_map[i].to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=(max(1, y2 - y1), max(1, x2 - x1)), mode='nearest')
    pred = pred.to(dtype=torch.int32)

    pred_coords[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    pred_coords[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    os.makedirs('./log_images',exist_ok=True)
    log = task.vq.log_images(sample['gt'], pred)
    for i in range(log['inputs'].shape[0]):
        vutils.save_image(log['inputs'][i], f'./log_images/{sample["id"][i]}_gt.png')
        vutils.save_image(log['reconstructions'][i], f'./log_images/{sample["id"][i]}.png')

    _confusion_matrix = task._calculate_confusion_matrix(pred.cpu().numpy(), sample['gt'].cpu().numpy())

    results = [
        {
         "uniq_id": sample_id,
         "bbox": pred_coords[i].tolist(),
         "image_code": hyps[i].tolist()[4:],
         "gt_code": sample['parsing_codes'][i].tolist(),
        }
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    return results, _confusion_matrix


def eval_refcoco(task, generator, models, sample, **kwargs):
    def _calculate_ap_score(hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    gen_out = task.inference_step(generator, models, sample)
    hyps = []
    for i in range(len(gen_out)):
        hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
    hyps = torch.stack(hyps, dim=0)
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    results = [
        {"uniq_id": sample_id,
         "box": [hyps[i][0].item(), hyps[i][1].item(), hyps[i][2].item(), hyps[i][3].item()]}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    scores = _calculate_ap_score(hyps, sample['region_coords'].float())
    return results, scores


def eval_snli_ve(task, generator, models, sample, **kwargs):
    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    eos_item = torch.tensor([task.src_dict.eos()])
    pad = task.src_dict.pad()
    valid_result = []
    for valid_answers, valid_constraint_masks in zip(task.valid_answers_list, task.valid_constraint_masks_list):
        valid_size = len(valid_answers)
        valid_tgt_items = [
            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_prev_items = [
            torch.cat([torch.tensor(decoder_prompt), valid_answer])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_constraint_mask_items = [
            torch.cat(
                [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask],
                dim=0
            )
            for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
        ]
        valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad).to(device)
        valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad).to(device)
        valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad).to(device)

        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    return results, scores


def eval_image_gen(task, generator, models, sample, **kwargs):
    hypos, _ = task.inference_image(generator, sample, models)
    tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
    caption = task.bpe.decode(task.tgt_dict.string([token for token in tokens if token >= 4]))[
              38:].replace('/', '')

    text_similarity_score, indices = task.compute_text_similarity(hypos, caption,
                                                                  sample['net_input']['src_tokens'].device)
    results = []
    for i, indice in enumerate(indices):
        results.append({"sample_id": str(sample["id"][0]), "score": text_similarity_score[i], "image": hypos[indice]})
    scores = [max(text_similarity_score).item()]
    sorted_hyps = [hypos[indice] for indice in indices]
    # dump results
    if task.cfg.gen_images_path:
        caption_tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
        caption = task.bpe.decode(task.tgt_dict.string([token for token in caption_tokens if token >= 4]))[
                  38:].replace('/', '')
        task.dump_images(sorted_hyps, text=caption, path=os.path.join(task.cfg.gen_images_path, 'all_results'))
        task.dump_images(sorted_hyps, text=caption, path=os.path.join(task.cfg.gen_images_path, 'top1'), topk=1)

    return results, scores


def eval_glue(task, generator, models, sample, **kwargs):
    net_output = models[0](**sample["net_input"])
    net_output[0].masked_fill_(~sample["constraint_masks"], -math.inf)
    last_token_ids = sample["net_input"]["prev_output_tokens"].ne(task.src_dict.pad()).sum(1, keepdim=True) - 1
    logits = net_output[0].gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, net_output[0].size(2)))
    logits = logits.squeeze(1)
    predicts = logits.argmax(1).tolist()
    hyps = [task.bpe.decode(task.src_dict[predict]).strip() for predict in predicts]
    results = [{"hyp": hyp, "ref": ref_dict.keys()[0]} for hyp, ref_dict in zip(hyps, sample['ref_dict'])]
    return results, None


def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == 'caption':
        return eval_caption(task, generator, models, sample, **kwargs)
    if task.cfg._name == 'rhrc':
        return eval_regioncaption(task, generator, models, sample, **kwargs)
    elif task.cfg._name == "caption_cn":
        return eval_caption_cn(task, generator, models, sample, **kwargs)
    elif task.cfg._name == "ocr":
        return eval_ocr(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'vqa_gen':
        return eval_vqa_gen(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'rkpt':
        return eval_refpose(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'rpar':
        return eval_refparsing(task, generator, models, sample, full_mask = False, **kwargs)
    elif task.cfg._name == 'rpar_full_mask':
        return eval_refparsing(task, generator, models, sample, full_mask = True, **kwargs)
    else:
        raise NotImplementedError


def merge_results(task, cfg, logger, score_cnt, score_sum, results):
    if task.cfg._name == 'image_gen':
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)
