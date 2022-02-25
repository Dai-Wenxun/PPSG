import torch
import random
import datetime
import numpy as np
import torch.nn.functional as F

from typing import List
from tasks import InputExample
from torch.nn import KLDivLoss


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%y-%m%d-%H%M')

    return cur


def beautify(args):
    args_info = '\n'
    args_info += f"method={args.method}\n"
    args_info += f"data_dir={args.data_dir}\n"
    args_info += f"model_name_or_path={args.model_name_or_path}\n"
    args_info += f"task_name={args.task_name}\n"
    args_info += f"max_length={args.max_length}\n"

    args_info += f"train_examples={args.train_examples}\n"
    args_info += f"dev_examples={args.dev_examples}\n"

    args_info += f"pattern_id={args.pattern_id}\n"
    args_info += f"per_gpu_train_batch_size={args.per_gpu_train_batch_size}\n"
    args_info += f"per_gpu_eval_batch_size={args.per_gpu_eval_batch_size}\n"
    args_info += f"per_gpu_unlabeled_batch_size={args.per_gpu_unlabeled_batch_size}\n"
    args_info += f"num_train_epochs={args.num_train_epochs}\n"
    args_info += f"gradient_accumulation_steps={args.gradient_accumulation_steps}\n"
    args_info += f"max_steps={args.max_steps}\n"
    args_info += f"logging_steps={args.logging_steps}\n"
    args_info += f"stopping_steps={args.stopping_steps}\n"
    args_info += f"repetitions={args.repetitions}\n"
    args_info += f"warmup_steps={args.warmup_steps}\n"
    args_info += f"learning_rate={args.learning_rate}\n"
    args_info += f"weight_decay={args.weight_decay}\n"
    args_info += f"adam_epsilon={args.adam_epsilon}\n"
    args_info += f"max_grad_norm={args.max_grad_norm}\n"
    args_info += f"seed={args.seed}\n"

    args_info += f"output_dir={args.output_dir}\n"
    args_info += f"label_list={args.label_list}\n"
    args_info += f"metrics={args.metrics}\n"
    args_info += f"device={args.device}\n"
    args_info += f"n_gpu={args.n_gpu}\n"
    args_info += f"label_smoothing={args.label_smoothing}\n"
    # args_info += f"={args.}\n"
    return args_info


def early_stopping(value, best, cur_step, max_step):
    stop_flag = False
    update_flag = False

    if value > best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
    return best, cur_step, stop_flag, update_flag


def distillation_loss(predictions, targets, temperature):
    """Compute the distillation loss (KL divergence between predictions and targets)"""
    # p = F.log_softmax(predictions / temperature, dim=1)
    p = F.log_softmax(predictions / temperature, dim=1)
    q = F.softmax(targets / temperature, dim=1)
    # q = targets

    # return KLDivLoss(reduction='sum')(p, q) * (temperature ** 2) / predictions.shape[0]
    return KLDivLoss(reduction='batchmean')(p, q)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def split_examples(examples: List[InputExample]) -> List[InputExample]:
    new_examples = []
    for example in examples:
        ex1 = InputExample(guid=example.guid, text_a=example.text_a, label=example.label)
        ex2 = InputExample(guid=example.guid, text_a=example.text_b, label=example.label)
        new_examples.append(ex1)
        new_examples.append(ex2)

    return new_examples
