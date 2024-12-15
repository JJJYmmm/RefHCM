"""
Precompute ngram counts of captions, to accelerate cider computation during training time.
"""

import os
import json
import argparse
import six
from six.moves import cPickle
from collections import defaultdict

import sys
from cider.cider_scorer import CiderScorer

import re
pattern = r'\s*\[\d+(?:, \d+)*\]\s*'

def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def get_doc_freq(refs, params):
    tmp = CiderScorer(df_mode="corpus")
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)

def pre_caption(caption):
    caption = re.sub(pattern, " ", caption)
    return caption

def build_dict(imgs, wtoi, params):
    wtoi['<eos>'] = 0

    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        #(params['split'] == 'val' and img['split'] == 'restval') or \
        ref_words = []
        ref_idxs = []
        caption = pre_caption(img[1]).strip().split()
        tmp_tokens = caption + ['<eos>']
        tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
        ref_words.append(' '.join(tmp_tokens))
        ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words, count_refs = get_doc_freq(refs_words, params)
    ngram_idxs, count_refs = get_doc_freq(refs_idxs, params)
    print('count_refs:', count_refs)
    return ngram_words, ngram_idxs, count_refs

def load_tsv(input_path):
    """
        return list[anno]
    """
    data_list = []
    with open(input_path, 'r') as file:
        for line in file:
            # 去除每行末尾的换行符，并按制表符分割成字段
            fields = line.strip().split('\t')
            data_list.append(fields)
    return data_list

def main(params):

    imgs = load_tsv(params['input_tsv'])
    dict_json = json.load(open(params['dict_json'], 'r'))
    itow = dict_json['ix_to_word']
    wtoi = {w:i for i,w in itow.items()}

    # Load bpe
    # if 'bpe' in dict_json:
    #     import tempfile
    #     import codecs
    #     codes_f = tempfile.NamedTemporaryFile(delete=False)
    #     codes_f.close()
    #     with open(codes_f.name, 'w') as f:
    #         f.write(dict_json['bpe'])
    #     with codecs.open(codes_f.name, encoding='UTF-8') as codes:
    #         bpe = apply_bpe.BPE(codes)
    #     params.bpe = bpe

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, params)

    pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','wb'))
    pickle_dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_tsv', default='data/dataset_coco.json', help='input tsv file to process into hdf5')
    parser.add_argument('--dict_json', default='data/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default='cihp_ferret_val', help='output pickle file')
    parser.add_argument('--split', default='all', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    main(params)