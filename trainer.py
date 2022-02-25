import copy
import os
import json
import torch
import numpy as np

import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import Union, Tuple
from logging import getLogger
from typing import List, Dict
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForSequenceClassification, BertTokenizer, BertForMaskedLM, BertModel
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import spearmanr, pearsonr

from random import uniform
from tasks import InputFeatures, DictDataset
from utils import early_stopping, distillation_loss
from preprocessor import SequenceClassifierPreprocessor, MLMPreprocessor, PromptPreprocessor
from model import BertForPromptClassification

logger = getLogger()

SEQ_CLS_TYPE = 'seq_cls'
MLM_TYPE = 'mlm'
MLM_PRETRAIN_TYPE = 'mlm_pretrain'


METHODS = [SEQ_CLS_TYPE, MLM_TYPE, MLM_PRETRAIN_TYPE]

PREPROCESSORS = {
    SEQ_CLS_TYPE: SequenceClassifierPreprocessor,
    MLM_TYPE: MLMPreprocessor,
    MLM_PRETRAIN_TYPE: PromptPreprocessor
}

EVALUATION_STEP_FUNCTIONS = {
    SEQ_CLS_TYPE: lambda trainer: trainer.seq_cls_eval_step,
    MLM_TYPE: lambda trainer: trainer.mlm_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    SEQ_CLS_TYPE: lambda trainer: trainer.seq_cls_train_step,
    MLM_TYPE: lambda trainer: trainer.mlm_train_step,
}

printed_flag = True


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)
        # self.writer = SummaryWriter(os.path.join(self.args.output_dir, 'runs'))
        self.preprocessor = PREPROCESSORS[self.args.method](self.args, self.tokenizer)
        self.train_step = TRAIN_STEP_FUNCTIONS[self.args.method](self)
        self.eval_step = EVALUATION_STEP_FUNCTIONS[self.args.method](self)

    def pre_train(self, train_examples: List[InputExample], checkpoint_path=None) -> Dict:
        self.init_model(checkpoint_path)
        train_dataset = self._generate_dataset(train_examples)
        train_sampler = RandomSampler(train_dataset)
        train_batch_size = self.args.per_gpu_unlabeled_batch_size * max(1, self.args.n_gpu)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * 5

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
        #                                             num_training_steps=t_total)

        # multi-gpu training
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        grad_acc_step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.zero_grad()

        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.to(self.args.device) for k, t in batch.items()}
                loss = self.mlm_pre_train_step(batch)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (grad_acc_step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    # scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        logger.info(json.dumps(logs))
                        if 0 < self.args.max_steps <= global_step:
                            epoch_iterator.close()
                            break
                grad_acc_step += 1
            if 0 < self.args.max_steps <= global_step:
                break

        self._save()
        self.model = None
        torch.cuda.empty_cache()

    def train(self, train_examples: List[InputExample], eval_examples: List[InputExample],
              unlabeled_examples: List[InputExample] = None, checkpoint_path=None) -> Dict:
        self.init_model(checkpoint_path)
        train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_dataset = self._generate_dataset(train_examples)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        unlabeled_dataloader, unlabeled_iter = None, None

        if unlabeled_examples:
            unlabeled_batch_size = self.args.per_gpu_unlabeled_batch_size * max(1, self.args.n_gpu)
            unlabeled_dataset = self._generate_dataset(unlabeled_examples)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size)
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // \
                                         (max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        if self.args.stopping_steps < 0:
            self.args.stopping_steps = t_total

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        # optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # scheduler = None
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        grad_acc_step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_res = {}

        best_score = -1.0
        stopping_step = 0
        stop_flag, update_flag = False, False

        self.model.zero_grad()

        logger.info(f"Scores before training {self.eval(eval_examples)['scores']}")

        for epoch in range(self.args.num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch:{epoch}:Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(self.args.device) for k, t in batch.items()}

                if unlabeled_dataloader:
                    while unlabeled_batch is None:
                        try:
                            unlabeled_batch = unlabeled_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting unlabeled dataset")
                            unlabeled_iter = unlabeled_dataloader.__iter__()
                    # lm_input_ids = unlabeled_batch['input_ids']
                    # unlabeled_batch['input_ids'], unlabeled_batch['mlm_labels'] = self._mask_tokens(lm_input_ids)
                    unlabeled_batch = {k: t.to(self.args.device) for k, t in unlabeled_batch.items()}

                loss = self.train_step(batch, unlabeled_batch)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (grad_acc_step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        logger.info(json.dumps(logs))

                        scores = self.eval(eval_examples)['scores']
                        logger.info(json.dumps(scores))
                        # self.writer.add_scalars('metrics', scores, global_step)
                        valid_score = scores[self.args.metrics[0]]
                        if self.args.stopping_steps > 0:
                            best_score, stopping_step, stop_flag, update_flag = early_stopping(
                                valid_score, best_score, stopping_step, max_step=self.args.stopping_steps)
                        else:
                            update_flag = True

                        if update_flag:
                            best_res = {'global_step': global_step, 'scores': scores}
                            self._save()

                        if stop_flag or 0 < self.args.max_steps <= global_step:
                            logger.info(best_res)
                            epoch_iterator.close()
                            break

                grad_acc_step += 1
            if stop_flag or 0 < self.args.max_steps <= global_step:
                break

        # if unlabeled_examples:
            # self._save()

        self.model = None
        torch.cuda.empty_cache()

        return best_res

    def _mask_tokens(self, input_ids):
        def _get_special_tokens_mask(tokenizer, token_ids_0):
            return list(
                map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] else 0,
                    token_ids_0))

        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, uniform(0.15, 0.55))
        # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
        #                        labels.tolist()]
        special_tokens_mask = [_get_special_tokens_mask(self.tokenizer, val) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        ignore_value = -100

        labels[~masked_indices] = ignore_value

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random].to(labels.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def eval(self, eval_examples: List[InputExample]) -> Dict:
        self.model.to(self.args.device)

        eval_dataset = self._generate_dataset(eval_examples)
        eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        results = {}
        preds, out_label_ids = None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(self.args.device) for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                logits = self.eval_step(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        results['logits'] = preds
        results['labels'] = out_label_ids
        results['predictions'] = np.argmax(results['logits'], axis=1)

        scores = {}
        for metric in self.args.metrics:
            if metric == 'acc':
                scores[metric] = accuracy_score(results['labels'], results['predictions'])
            elif metric == 'mathws':
                scores[metric] = matthews_corrcoef(results['labels'], results['predictions'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], results['predictions'])
            elif metric == 'prson':
                scores[metric] = pearsonr(results['labels'], results['logits'].reshape(-1))[0]
            elif metric == 'sprman':
                scores[metric] = spearmanr(results['labels'], results['logits'].reshape(-1))[0]
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        results['scores'] = scores

        return results

    def mlm_pre_train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs_a = {'input_ids': batch['input_ids_x'], 'attention_mask': batch['attention_mask_x'],
                    'token_type_ids': batch['token_type_ids_x']}
        logits_a = self.model(**inputs_a)[0]
        logits_a = logits_a[batch['mlm_labels_x'] >= 0]

        inputs_b = {'input_ids': batch['input_ids_y'], 'attention_mask': batch['attention_mask_y'],
                    'token_type_ids': batch['token_type_ids_y']}
        logits_b = self.model(**inputs_b)[0]
        logits_b = logits_b[batch['mlm_labels_y'] >= 0]

        cos_sim = nn.CosineSimilarity(dim=-1)(logits_a.unsqueeze(1), logits_b.unsqueeze(0)) / 0.05
        loss_fct = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)

        loss = loss_fct(cos_sim, labels)

        return loss

    def mlm_train_step(self, batch: Dict[str, torch.Tensor],
                       unlabeled_batch: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        inputs = self._generate_default_inputs(batch)
        mlm_labels, labels = batch['mlm_labels'], batch['labels']

        # input_ids = inputs.pop('input_ids')
        # inputs_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
        # outputs = self.model(inputs_embeds=inputs_embeds, **inputs)

        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])

        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.args.label_list)), labels.view(-1))

        if self.args.label_smoothing:
            label_smoothing_loss = distillation_loss(prediction_scores, batch['logits'], 20.)
            # print(f"label_smoothing_loss:{label_smoothing_loss}")
            # print(f"raw_loss:{loss}")

            loss = loss + label_smoothing_loss / (label_smoothing_loss / loss).detach()

        if unlabeled_batch:
            # lm_inputs = self._generate_default_inputs(unlabeled_batch)
            # lm_inputs['labels'] = unlabeled_batch['mlm_labels']
            # lm_loss = self.model(**lm_inputs)[0]
            # loss = self.args.alpha * loss + (1 - self.args.alpha) * lm_loss

            distill_inputs = self._generate_default_inputs(unlabeled_batch)
            distill_outputs = self.model(**distill_inputs)
            logits_predicted = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(unlabeled_batch['mlm_labels'], distill_outputs[0])
            logits_target = unlabeled_batch['logits']

            distill_loss = distillation_loss(logits_predicted, logits_target, 20.)
            # logger.info(distill_loss)
            loss = 0.5 * distill_loss + 0.5 * loss

        return loss

    def seq_cls_train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier training step."""
        inputs = self._generate_default_inputs(batch)
        inputs['labels'] = batch['labels']
        outputs = self.model(**inputs)

        return outputs[0]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self._generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def seq_cls_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self._generate_default_inputs(batch)
        return self.model(**inputs)[0]

    def _generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'],
                  'token_type_ids': batch['token_type_ids']}
        return inputs

    def _generate_dataset(self, examples: List[InputExample]):
        features = self._convert_examples_to_features(examples)
        if isinstance(features[0], tuple):
            feature_dict = {
                'input_ids_x': torch.tensor([f[0].input_ids for f in features], dtype=torch.long),
                'attention_mask_x': torch.tensor([f[0].attention_mask for f in features], dtype=torch.long),
                'token_type_ids_x': torch.tensor([f[0].token_type_ids for f in features], dtype=torch.long),
                'mlm_labels_x': torch.tensor([f[0].mlm_labels for f in features], dtype=torch.long),

                'input_ids_y': torch.tensor([f[1].input_ids for f in features], dtype=torch.long),
                'attention_mask_y': torch.tensor([f[1].attention_mask for f in features], dtype=torch.long),
                'token_type_ids_y': torch.tensor([f[1].token_type_ids for f in features], dtype=torch.long),
                'mlm_labels_y': torch.tensor([f[1].mlm_labels for f in features], dtype=torch.long)
            }
        else:
            feature_dict = {
                'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
                'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                'labels': torch.tensor([f.label for f in features]),  # might be float
                'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
                'logits': torch.tensor([f.logits for f in features], dtype=torch.float)
            }

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        global printed_flag
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.preprocessor.get_input_features(example)
            features.append(input_features)
            if ex_index < 1 and printed_flag:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
                printed_flag = False
        return features

    def _save(self) -> None:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.saved_path)
        self.tokenizer.save_pretrained(self.args.saved_path)
        logger.info(f"Model saved at {self.args.saved_path}")

    def init_model(self, checkpoint_path=None):
        if checkpoint_path:
            model_name_or_path = checkpoint_path
        else:
            model_name_or_path = self.args.model_name_or_path

        if self.args.method in SEQ_CLS_TYPE:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=len(self.args.label_list)).to(self.args.device)
        elif self.args.method == MLM_TYPE:
            self.model = BertForMaskedLM.from_pretrained(model_name_or_path).to(self.args.device)
        elif self.args.method == MLM_PRETRAIN_TYPE:
            self.model = BertModel.from_pretrained(model_name_or_path).to(self.args.device)

        logger.info(f'Load parameters from {model_name_or_path}')

    def _generate_unlabeled_dataset(self, examples):
        inputs = self.tokenizer(
            [(ex.text_a, ex.text_b) for ex in examples],
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        logger.info(self.tokenizer.decode(inputs['input_ids'][0]))
        return DictDataset(**inputs)
