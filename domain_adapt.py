import os
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from typing import List
from logging import getLogger
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks import DictDataset, InputExample
from preprocessor import MLMAdaptPreprocessor, PromptMLMAdaptPreprocessor

logger = getLogger()


MLM_ADAPT = 'mlm_adapt'
PROMPT_ADAPT = 'prompt_mlm_adapt'

ADAPT_METHODS = [MLM_ADAPT, PROMPT_ADAPT]

PREPROCESSORS = {
    MLM_ADAPT: lambda args, tokenizer: MLMAdaptPreprocessor(args, tokenizer),
    PROMPT_ADAPT: lambda args, tokenizer: PromptMLMAdaptPreprocessor(args, tokenizer)
}


class AdaptConfig:
    method = None
    data_dir = None
    output_dir = None
    model_name_or_path = None
    task_name = None
    max_length = None
    device = None
    n_gpu = None
    pattern_id = None
    per_gpu_train_batch_size = 64
    gradient_accumulation_steps = 1
    max_steps = -1
    num_train_epochs = 10
    logging_steps = 50
    warmup_steps = 0
    learning_rate = 5e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    mask_ratio = 0.15

    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


class AdaptTrainer:
    def __init__(self,
                 method,
                 data_dir,
                 output_dir,
                 model_name_or_path,
                 task_name,
                 max_length,
                 device, 
                 n_gpu,
                 pattern_id
                 ):
        self.args = AdaptConfig(locals())
        self._print_hyper_params()

        self.model = BertForMaskedLM.from_pretrained(model_name_or_path).to(device)

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.writer = SummaryWriter(output_dir)
        self.preprocessor = PREPROCESSORS[method](self.args, self.tokenizer)

    def train(self, train_examples):
        train_dataset = self._generate_dataset(train_examples)

        train_sampler = RandomSampler(train_dataset)
        train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // \
                                         (max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

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
                loss = self._mlm_train_step(batch)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (grad_acc_step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
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
                        self.writer.add_scalar('loss', loss_scalar, global_step)
                        if 0 < self.args.max_steps <= global_step:
                            epoch_iterator.close()
                            break
                grad_acc_step += 1
            if 0 < self.args.max_steps <= global_step:
                break

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        logger.info(f"Model saved at {self.args.output_dir}")

        # Clear cache
        self.model = None
        torch.cuda.empty_cache()

    def _mlm_train_step(self, batch):
        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'])
        loss = self.model(**batch)[0]
        return loss

    def _mask_tokens(self, input_ids):
        def _get_special_tokens_mask(tokenizer, token_ids_0):
            return list(
                map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] else 0,
                    token_ids_0))

        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, self.args.mask_ratio)
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

    def _print_hyper_params(self):
        args_info = '\n'
        args_info += f"method={self.args.method}\n"
        args_info += f"per_gpu_train_batch_size={self.args.per_gpu_train_batch_size}\n"
        args_info += f"n_gpu={self.args.n_gpu}\n"
        args_info += f"gradient_accumulation_steps={self.args.gradient_accumulation_steps}\n"
        args_info += f"max_steps={self.args.max_steps}\n"
        args_info += f"num_train_epochs={self.args.num_train_epochs}\n"
        args_info += f"warmup_steps={self.args.warmup_steps}\n"
        args_info += f"learning_rate={self.args.learning_rate}\n"
        args_info += f"weight_decay={self.args.weight_decay}\n"
        args_info += f"adam_epsilon={self.args.adam_epsilon}\n"
        args_info += f"max_grad_norm={self.args.max_grad_norm}\n"
        args_info += f"mask_ratio={self.args.mask_ratio}\n"
        args_info += f"pattern_id={self.args.pattern_id}\n"
        logger.info(args_info)

    def _generate_dataset(self, examples: List[InputExample]):
        features = []
        for ex_index, example in enumerate(examples):
            feature = self.preprocessor.get_input_features(example)
            features.append(feature)
            if ex_index < 1:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(feature.pretty_print(self.tokenizer))
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        }

        return DictDataset(**feature_dict)
