import os
import json
import copy
import re
import torch
import numpy as np
import random
import string
from tqdm import tqdm
from utils import *
import T5_gen

import argparse


def line_tokenizer(text):
    return text.split()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class FewGLUEAug:
    @staticmethod
    def read_jsonl(file_path):
        examples = []
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                examples.append(example_json)
        return examples

    @staticmethod
    def save_jsonl(examples, save_path):
        with open(save_path, "w", encoding="utf8") as f:
            for e in examples:
                f.write(json.dumps(e) + '\n')
        f.close()

    @staticmethod
    def mask_text(text, mask_ratio=0.5, cnt=0, allow_substitute_punctuation=False):
        substitute_verbalizers = ['<extra_id_{}>'.format(i) for i in range(100)]
        tokens = nltk_line_tokenizer(text)
        n = len(tokens)

        if allow_substitute_punctuation:
            indices = sorted(random.sample(range(n), int(n * mask_ratio)))
        else:
            candidate_idxs = [i for i in range(n) if tokens[i] not in string.punctuation]
            n = len(candidate_idxs)
            indices = sorted(random.sample(candidate_idxs, int(n * mask_ratio)))

        if len(indices) == 0:
            indices = sorted(random.sample(range(n), 1))

        masked_src, masked_tgt = "", []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[idx]
            tokens[idx] = "[MASK]"

        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
                continue
            if token == "[MASK]":
                masked_src += " " + substitute_verbalizers[cnt]
                cnt += 1
            else:
                masked_src += " " + token
        return masked_src.strip(), cnt

    @staticmethod
    def predict_blanks(texts_to_be_augmented, gen_blanks_func, aug_kwargs):
        total_predictions, pred_blanks = gen_blanks_func(texts_to_be_augmented, **aug_kwargs)
        pred_blanks = [pred_blank[0] for pred_blank in pred_blanks]
        # pred_blanks=pred_blanks[0]
        return pred_blanks

    @staticmethod
    def recover_examples_from_blanks(pure_parts, pred_blanks):
        # example_lines=[['[MASK] x','[MASK] y'],['x [MASK] y', '[MASK] z']]
        # pred_blanks=[['a','b'],['c','d']]
        # return filled_parts=[['a x','b y'],['x c y','d z']]
        filled_parts = []
        for (parts, pred_blank) in zip(pure_parts, pred_blanks):
            current_blank = 0
            filled_parts.append([])
            for part in parts:
                output_tokens = []
                tokens = part.split()
                for token in tokens:
                    if token.startswith('<extra_id_'):
                        if current_blank < len(pred_blank):
                            output_tokens.append(pred_blank[current_blank])
                        current_blank += 1
                    else:
                        output_tokens.append(token)
                filled_parts[-1].append(' '.join((' '.join(output_tokens)).split()).strip())
        return filled_parts

    @staticmethod
    def postprocess_texts(filled_parts):
        processed_parts = []
        for parts in filled_parts:
            processed_parts.append([])
            for part in parts:
                processed_parts[-1].append(part.strip(string.punctuation).strip())
        return processed_parts


class SSt2Aug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug, self).__init__()
        self.TASK_NAME = "sst-2"

    def aug_with_pattern(self, sentence, label, gen_blanks_func, aug_kwargs,
                         label_type='flip', mask_ratio=0.5, aug_num=1):
        bad_words_ids = [[3], [19794], [22354]] + [[2163], [4273], [465], [150], [1525], [58]]
        aug_kwargs['bad_words_ids'] = bad_words_ids
        texts_to_be_augmented = []
        masked_parts = []
        new_sentences = []
        new_labels = []
        for aug_idx in range(aug_num):
            if label_type == 'flip':
                if label == '1':
                    label_text = 'No'
                    new_label = '0'
                elif label == '0':
                    label_text = "Yes"
                    new_label = "1"
            else:
                if label == '1':
                    label_text = 'Yes'
                elif label == '0':
                    label_text = "No"
                new_label = label

            new_labels.append(new_label)
            masked_sentence, cnt = self.mask_text(sentence, mask_ratio=mask_ratio)

            texts_to_be_augmented.append(masked_sentence + '?' + label_text + '.')
            masked_parts.append([masked_sentence])

        pred_blanks = self.predict_blanks(texts_to_be_augmented, gen_blanks_func, aug_kwargs)
        filled_parts = self.recover_examples_from_blanks(masked_parts, pred_blanks)
        filled_parts = self.postprocess_texts(filled_parts)
        for parts in filled_parts:
            [new_sentence] = parts
            new_sentences.append(new_sentence)
        return new_sentences, new_labels

    def augment(self, data_path, aug_func, aug_func_name, aug_kwargs):
        examples = self.read_jsonl(os.path.join(data_path, "{}/train200.jsonl".format(self.TASK_NAME)))
        new_examples = []
        for e in tqdm(examples):
            new_sentence, new_label = self.aug_with_pattern(e["sentence"], e["label"], aug_func, **aug_kwargs)

            for (x, y) in zip(new_sentence, new_label):
                tmp_e = copy.deepcopy(e)
                tmp_e["sentence"] = x
                tmp_e["label"] = y
                tmp_e["orig_label"] = e["label"]
                new_examples.append(tmp_e)

        if not os.path.exists(os.path.join(data_path, "augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path, "augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,
                        os.path.join(data_path, "augmented/{}/{}_train200.jsonl".format(self.TASK_NAME, aug_func_name)))


FEWGLUEAUGS = {
    "sst-2": SSt2Aug
}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--mask_ratio", required=True, type=float)
    parser.add_argument("--label_type", type=str, default="flip")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--aug_num", type=int, default=150)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = init_args()
    aug = FEWGLUEAUGS[args.task_name]()
    data_path = "data/FewGLUE_dev32/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    set_seed(1)
    t5aug = gen_aug_T5.T5Aug()
    aug_func = t5aug.generate_blanks
    aug_func_name = f't5_{args.label_type}_{args.mask_ratio}_sample{int(args.do_sample)}' \
                    f'_beam{args.num_beams}_augnum{args.aug_num}'

    aug_kwargs = {'label_type': args.label_type, 'mask_ratio': args.mask_ratio, 'aug_num': args.aug_num,
                  'aug_kwargs': {'do_sample': args.do_sample, 'num_beams': args.num_beams}}
    print(aug_kwargs)
    aug.augment(data_path, aug_func, aug_func_name, aug_kwargs)
