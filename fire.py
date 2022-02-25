import os
import torch
import argparse
from logging import getLogger

from logger import init_logger
from tasks import PROCESSORS, METRICS, DEFAULT_METRICS
from utils import beautify, get_local_time
from modeling import adapt_train, baseline_train, train_final_model, train_single_model


METHODS = ['MLM', 'FT', 'PT']


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="The training method to use.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--max_length", default=None, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    # dataset parameters
    parser.add_argument("--train_examples", default=0.01, type=float,
                        help="<= 1 means the ratio to total train examples, > 1 means the number of train examples.")
    parser.add_argument("--dev_examples", default=1.0, type=float,
                        help="<= 1 means the ratio to total train examples, > 1 means the number of train examples.")

    # training & evaluation parameters
    parser.add_argument("--pattern_id", default=0, type=int,
                        help="The ids of the PVPs to be used")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--stopping_steps', type=int, default=-1,
                        help="Early stopping steps")
    parser.add_argument('--repetitions', default=3, type=int,
                        help="The number of times to repeat training and testing with different seeds.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=list, default=[0, 10, 100],
                        help="random seed for initialization")
    parser.add_argument('--label_smoothing', type=bool, default=True,
                        help="random seed for initialization")
    args = parser.parse_args()
    args.output_dir = os.path.join('./output', args.task_name, args.model_name_or_path.split('/')[-1], get_local_time())

    # Init logger
    init_logger(args.output_dir)
    logger = getLogger()

    # Parameters addition
    args.label_list = PROCESSORS[args.task_name]().get_labels()
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()

    logger.info("Parameters: {}".format(beautify(args)))

    if args.method == 'MLM':
        adapt_train(args)
    elif args.method == 'FT':
        baseline_train(args)
    elif args.method == 'PT':
        train_final_model(args)


if __name__ == '__main__':
    main()
