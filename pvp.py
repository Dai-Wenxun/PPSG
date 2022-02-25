# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import string
from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import torch
from transformers import BertTokenizer
from logging import getLogger
from tasks import InputExample

logger = getLogger()

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    def __init__(self, args, tokenizer: BertTokenizer, pattern_id):
        self.args = args
        self.tokenizer = tokenizer
        self.pattern_id = pattern_id
        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        return self.tokenizer.mask_token_id

    @staticmethod
    def shortenable(s):
        return s, True

    @staticmethod
    def remove_final_punc(s: str):
        return s.rstrip(string.punctuation)

    def encode(self, example: InputExample) -> Tuple[List[int], List[int]]:

        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(self.tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(self.tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]]):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - self.args.max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def _build_mlm_logits_to_cls_logits_tensor(self):
        if not hasattr(self.args, 'label_list') or self.args.method == 'mlm_pretrain':
            return
        label_list = self.args.label_list
        m2c_tensor = torch.ones(len(label_list), dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizer = self.verbalize(label)
            verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
            m2c_tensor[label_idx] = verbalizer_id
        return m2c_tensor

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        cls_logits = logits[m2c]
        return cls_logits  # cls_logits.shape() == num_labels


class ColaPVP(PVP):
    VERBALIZER = [
        {'0': 'proof', '1': 'one'},
        {'0': 'sad', '1': 'wrong'},
        {'0': 'disappointing', '1': 'misleading'},
        {'0': 'no', '1': 'yes'}
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return [text_a, 'You are ', self.mask, '.'], []
        elif self.pattern_id == 1:
            return ['It is ', self.mask, '.', text_a], []
        elif self.pattern_id == 2:
            return ['I am ', self.mask, '.', text_a], []
        elif self.pattern_id == 3:
            return [self.shortenable(self.remove_final_punc(text_a[0])), '?', self.mask, '.'], []

    def verbalize(self, label) -> str:
        return ColaPVP.VERBALIZER[self.pattern_id][label]


class MnliPVP(PVP):
    VERBALIZER = [
        {"entailment": "Fine", "neutral": "Plus", "contradiction": "Otherwise"},
        {"entailment": "There", "neutral": "Plus", "contradiction": "Otherwise"},
        {"entailment": "Meaning", "neutral": "Plus", "contradiction": "Otherwise"},
        {"entailment": "Yes", "neutral": "Maybe", "contradiction": "No"},
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '.'], [self.mask, ', you are right ,', text_b]
        elif self.pattern_id == 1:
            return [text_a, '.'], [self.mask, 'you’re right', text_b]
        elif self.pattern_id == 2:
            return [text_a, '.'], [self.mask, "!", text_b]
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> str:
        return MnliPVP.VERBALIZER[self.pattern_id][label]


class MrpcPVP(PVP):
    VERBALIZER = [
        {"0": "Alas", "1": "Rather"},
        {"0": "Thus", "1": "At"},
        {"0": "Moreover", "1": "Instead"},
        {"0": 'No', "1": "Yes"}
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '.'], [self.mask, '!', text_b]
        elif self.pattern_id == 1:
            return [text_a, '.'], [self.mask, '. This is the first time', text_b]
        elif self.pattern_id == 2:
            return [text_a, '.'], [self.mask, ". That's right .", text_b]
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> str:
        return MrpcPVP.VERBALIZER[self.pattern_id][label]


class Sst2PVP(PVP):
    VERBALIZER = [
        {'0': 'pathetic', '1': 'irresistible'},
        {'0': 'bad', '1': 'wonderful'},
        {'0': 'bad', '1': 'delicious'},
        {'0': 'no', '1': 'yes'}
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))

        if self.pattern_id == 0:
            return [text_a, '. A', self.mask, 'one .'], []
        elif self.pattern_id == 1:
            # return [text_a, '. Is bad or wonderful ?', 'It is ', self.mask, '.'], []
            return [text_a, '. ', 'It is ', self.mask, '.'], []

            # return [text_a, '. A', self.mask, 'piece .'], []
        elif self.pattern_id == 2:
            return [text_a, '. All in all', self.mask, '.'], []
        elif self.pattern_id == 3:
            return [text_a, '?', self.mask, '.'], []

    def verbalize(self, label) -> str:
        return Sst2PVP.VERBALIZER[self.pattern_id][label]


class QqpPVP(PVP):
    VERBALIZER = [
        {"0": "Since", "1": "Me"},
        {"0": "Best", "1": "Um"},
        {"0": "Beyond", "1": "Ironically"},
        {"0": 'No', "1": "Yes"}
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '?'], [self.mask, ', but', text_b]
        elif self.pattern_id == 1:
            return [text_a, '?'], [self.mask, ', please ,', text_b]
        elif self.pattern_id == 2:
            return [text_a, '?'], [self.mask, ", I want to know", text_b]
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> str:
        return QqpPVP.VERBALIZER[self.pattern_id][label]


class QnliPVP(PVP):
    VERBALIZER = [
        {"entailment": "Okay", "not_entailment": "Nonetheless"},
        {"entailment": "Notably", "not_entailment": "Yet"},
        {"entailment": "Specifically", "not_entailment": "Notably"},
        {"entailment": "Yes", "not_entailment": "No"},
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '?'], [self.mask, '. Yes ,', text_b]
        elif self.pattern_id == 1:
            return [text_a, '?'], [self.mask, '. It is known that', text_b]
        elif self.pattern_id == 2:
            return [text_a, '?'], [self.mask, ", however ,", text_b]
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> str:
        return QnliPVP.VERBALIZER[self.pattern_id][label]


class RtePVP(PVP):
    VERBALIZER = [
        {"not_entailment": "Yet", "entailment": "Clearly"},
        {"not_entailment": "meanwhile", "entailment": "Accordingly"},
        {"not_entailment": "Meanwhile", "entailment": "So"},
        {"not_entailment": "no", "entailment": "yes"},
    ]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '.'], [self.mask, ', I believe', text_b]
        elif self.pattern_id == 1:
            return [text_a, '.'], [self.mask, ', I think that', text_b]
        elif self.pattern_id == 2:
            return [text_a, '.'], [self.mask, ", I think", text_b]
        elif self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> str:
        return RtePVP.VERBALIZER[self.pattern_id][label]


class SSt2PromptPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:
        if self.pattern_id == 0:
            return [self.shortenable(example.text_a)], ['. A', self.mask, 'piece .']
        elif self.pattern_id == 1:
            return [self.shortenable(example.text_a), '. A', self.mask, 'piece .'], []

    def verbalize(self, label) -> List[str]:
        pass


class PromptPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:
        if example.text_b is None:
            if self.pattern_id == 0:
                return ['This sentence : "', self.shortenable(example.text_a),
                        '" means ', self.tokenizer.mask_token, '.'], []
            elif self.pattern_id == 1:
                return ['This sentence of "', self.shortenable(example.text_a),
                        '" means ', self.tokenizer.mask_token, '.'], []
        else:
            if self.pattern_id == 0:
                return ['This sentence : "', self.shortenable(example.text_a), '?'], \
                       [self.shortenable(example.text_b), '" means ', self.tokenizer.mask_token, '.']
            elif self.pattern_id == 1:
                return ['This sentence of "', self.shortenable(example.text_a), '?'], \
                       [self.shortenable(example.text_b), '" means ', self.tokenizer.mask_token, '.']

    def verbalize(self, label) -> List[str]:
        pass

PVPS = {
    'mnli': MnliPVP,
    'cola': ColaPVP,
    'sst-2': Sst2PVP,
    'mrpc': MrpcPVP,
    'qqp': QqpPVP,
    'qnli': QnliPVP,
    'rte': RtePVP
}
