import copy
import os
from logging import getLogger
from numpy import mean, std
import numpy as np
from trainer import Trainer
from utils import set_seed
from tasks import load_examples, DEV_SET, TRAIN_SET
from domain_adapt import AdaptTrainer

logger = getLogger()


def logger_helper(results, metrics):
    for res in results:
        logger.info(res)
    avg_scores = {metric: [] for metric in metrics}
    for rp in range(len(results)):
        for metric in metrics:
            avg_scores[metric].append(results[rp]['scores'][metric] * 100)

    logger.info([f"avg_{metric}': {round(mean(avg_scores[metric]), 3)}, "
                 f"std_{metric}: {round(std(avg_scores[metric]), 2)}" for metric in metrics])


def entropy(x):
    return np.sum(np.log(x) * x)


def select_all_examples(unlabeled_examples, trainer, checkpoint_path):
    trainer.init_model(checkpoint_path=checkpoint_path)
    results = trainer.eval(unlabeled_examples)

    candidates = []
    for idx, (label, scores) in enumerate(zip(results['predictions'], results['logits'])):
        probs = np.exp(scores) / np.sum(np.exp(scores))
        candidates.append((idx, label, probs))

    # candidates.sort(key=lambda e: e[0], reverse=True)
    # print(candidates[:5])

    label_list = ["0", "1"]
    returns = []
    for idx, label, probs in candidates:
        unlabeled_examples[idx].label = label_list[label]
        unlabeled_examples[idx].logits = probs
        returns.append(unlabeled_examples[idx])

    print("cur size:{}".format(len(returns)))
    return returns


def select_most_confident_examples(unlabeled_examples, trainer, checkpoint_path):
    trainer.init_model(checkpoint_path=checkpoint_path)
    results = trainer.eval(unlabeled_examples)

    candidates = []
    for idx, (label, scores) in enumerate(zip(results['predictions'], results['logits'])):
        probs = np.exp(scores) / np.sum(np.exp(scores))
        candidates.append((idx, label, probs))

    # candidates.sort(key=lambda e: e[2][e[1]], reverse=True)
    candidates.sort(key=lambda e: entropy(e[2]), reverse=True)

    print(candidates[:5])

    label_list = ["0", "1"]
    returns = []
    for idx, label, probs in candidates[:20000]:
        # if probs[label] < 0.99:
        #     continue
        unlabeled_examples[idx].label = label_list[label]
        unlabeled_examples[idx].logits = probs
        returns.append(unlabeled_examples[idx])

    print("cur size:{}".format(len(returns)))
    return returns


def select_aug_examples(trainer, checkpoint_path, seed):
    trainer.init_model(checkpoint_path=checkpoint_path)
    from tasks import InputExample
    from test import read_jsonl
    path = './FlipDA-main/genaug/data/FewGLUE_dev32/augmented/sst-2'
    flip_path = os.path.join(path, f't5_flip_0.8_sample0_beam1_augnum150_train{seed}.jsonl')
    exs = read_jsonl(flip_path)
    processed_exs = []
    for ex in exs:
        processed_exs.append(InputExample(0, ex['sentence'], label=ex['label']))

    # return processed_exs

    results = trainer.eval(processed_exs)

    returns = []
    done_texta = {}

    label_list = ["0", "1"]
    for idx, (label, scores) in enumerate(zip(results['predictions'], results['logits'])):
        probs = np.exp(scores) / np.sum(np.exp(scores))
        if processed_exs[idx].text_a:
            #  and not processed_exs[idx].text_a in done_texta
            if label_list[label] == processed_exs[idx].label:
                returns.append(processed_exs[idx])
            else:
                processed_exs[idx].label = label_list[label]
                returns.append(processed_exs[idx])

            done_texta[processed_exs[idx].text_a] = True

        # label_index = probs.argmax()
        # if probs[label_index] > 0.99:
        #     processed_exs[idx].label = label_list[label_index]
        #     returns.append(processed_exs[idx])

        # if label_list[label] != processed_exs[idx].label:
        #     if processed_exs[idx].text_a:
        #         processed_exs[idx].label = label_list[label]
        #         returns.append(processed_exs[idx])

    print("cur size:{}".format(len(returns)))
    return returns


def discuss_aug_examples(trainer, checkpoint_path, seed):
    trainer.init_model(checkpoint_path=checkpoint_path)
    from tasks import InputExample
    from test import read_jsonl, save_jsonl
    path = './FlipDA-main/genaug/data/FewGLUE_dev32/augmented/SST2'
    flip_path = os.path.join(path, f't5_flip_0.8_sample0_beam1_augnum50_train{seed}.jsonl')
    exs = read_jsonl(flip_path)
    processed_exs = []
    for ex in exs:
        processed_exs.append(InputExample(0, ex['sentence'], label=ex['label']))

    # return processed_exs

    results = trainer.eval(processed_exs[:5000])

    returns = []
    label_list = {"0": 0, "1": 1}
    for idx, (label, scores) in enumerate(zip(results['predictions'], results['logits'])):
        probs = np.exp(scores) / np.sum(np.exp(scores))
        exs[idx]['probs'] = str(probs[label_list[exs[idx]['label']]])

    save_jsonl(exs, './test.jsonl')

    print("cur size:{}".format(len(returns)))
    return returns


def train_final_model(args, checkpoint_path='output/sst-2/bert-base-uncased/test0'):
    all_results = []
    final_results = []
    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                               num_examples=1. - args.train_examples, seed=domain_seed)
        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)
            trainer = Trainer(args)
            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)
            logger.info("final_training start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'final_training')
            # pretrained_path = os.path.join(checkpoint_path, info, "final_training")
            pretrained_path = os.path.join(checkpoint_path, info, "final_training")
            # while init_size != len(unlabeled_examples):
            #     trainer.train(more_exs+fine_tune_examples, eval_examples=eval_examples, checkpoint_path=pretrained_path)
            #     more_exs = select_most_confident_examples(init_size, unlabeled_examples, trainer, args.saved_path)
            #
            #     pretrained_path = args.saved_path

            # args.max_steps = -1
            # args.num_train_epochs = 1

            # discuss_aug_examples(trainer, pretrained_path, domain_seed)

            # more_exs = select_aug_examples(trainer, pretrained_path, domain_seed)

            # ex_list = []
            # for ex in more_exs:
            #     ex_dict = {}
            #     ex_dict['sentence'] = ex.text_a
            #     ex_dict['label'] = ex.label
            #     ex_list.append(ex_dict)
            #
            # from test import save_jsonl
            # save_jsonl(ex_list, './100.jsonl')

            # args.learning_rate *= 0.1

            # if len(more_exs) > 50000:
            #     args.max_steps = -1
            #     args.num_train_epochs = 1
            # else:
            #     args.max_steps = 5000
            #
            # args.warmup_steps = 2000
            #
            # # args.num_train_epochs = 1
            # final_results.append(trainer.train(more_exs + fine_tune_examples,
            #                                    eval_examples=eval_examples,
            #                                    # unlabeled_examples=more_exs,
            #                                    checkpoint_path=pretrained_path))
            temp = []
            for i in range(3):
                args.warmup_steps = 2000
                args.max_steps = -1
                args.num_train_epochs = 1
                # more_exs = select_most_confident_examples(unlabeled_examples, trainer, pretrained_path)
                more_exs = select_all_examples(unlabeled_examples, trainer, pretrained_path)
                # more_exs = select_aug_examples(trainer, pretrained_path)

                temp.append(trainer.train(more_exs + fine_tune_examples,
                                          eval_examples=eval_examples,
                                          # unlabeled_examples=more_exs,
                                          checkpoint_path=pretrained_path))
                logger.info(temp)
                pretrained_path = args.saved_path

            all_results += temp
            temp.sort(key=lambda e: e['scores']['acc'], reverse=True)
            final_results.append(temp[0])

    logger.info('all_results:')
    logger_helper(all_results, args.metrics)
    logger.info('final_results:')
    logger_helper(final_results, args.metrics)


def train_single_model(args, checkpoint_path='output/sst-2/bert-base-uncased'):
    fine_tune_with_lm_training_results = []
    fine_tune_with_lm_training_and_mlm_adapted_results = []
    fine_tune_with_lm_training_and_prompt_mlm_adapted_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    for domain_repetition in range(args.repetitions):

        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                                               num_examples=1. - args.train_examples, seed=domain_seed)

        pretrained_path = os.path.join(checkpoint_path, 'MLM_Adapt', f'Seed-{domain_seed}')
        pretrained_path_prompt = os.path.join(checkpoint_path, 'P-MLM_Adapt', f'Seed-{domain_seed}')

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)

            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)

            trainer = Trainer(args)
            logger.info("fine_tune_with_lm_training start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'test')
            args.max_steps = -1
            args.num_train_epochs = 2
            fine_tune_with_lm_training_results.append(
                trainer.train(fine_tune_examples + unlabeled_examples,
                              eval_examples=eval_examples,
                              # unlabeled_examples=unlabeled_examples
                              ))

            # logger.info("fine_tune_with_lm_training_and_mlm_adapted start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_lm_training_and_mlm_adapted')
            # fine_tune_with_lm_training_and_mlm_adapted_results.append(
            #     trainer.train(fine_tune_examples,
            #                   eval_examples=eval_examples,
            #                   # unlabeled_examples=unlabeled_examples,
            #                   checkpoint_path=pretrained_path))

            # logger.info("fine_tune_with_lm_training_and_prompt_mlm_adapted start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_lm_training_and_prompt_mlm_adapted')
            # fine_tune_with_lm_training_and_prompt_mlm_adapted_results.append(
            #     trainer.train(fine_tune_examples, eval_examples=eval_examples,
            #                   unlabeled_examples=unlabeled_examples,
            #                   checkpoint_path=pretrained_path_prompt))

    logger.info('fine_tune_with_lm_training_results:')
    logger_helper(fine_tune_with_lm_training_results, args.metrics)
    logger.info('fine_tune_with_lm_training_and_mlm_adapted_results:')
    logger_helper(fine_tune_with_lm_training_and_mlm_adapted_results, args.metrics)
    logger.info('fine_tune_with_lm_training_and_prompt_mlm_adapted_results:')
    logger_helper(fine_tune_with_lm_training_and_prompt_mlm_adapted_results, args.metrics)


def baseline_train(args, checkpoint_path='output/sst-2/bert-base-uncased/'):
    fine_tune_results = []
    fine_tune_with_mlm_adapted_results = []
    fine_tune_with_prompt_mlm_adapted_results = []

    eval_examples, _ = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=args.dev_examples)

    args.repetitions = 3
    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        _, fine_tune_examples = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                              num_examples=1. - args.train_examples, seed=domain_seed)

        pretrained_path = os.path.join(checkpoint_path, 'MLM_Adapt', f'Seed-{domain_seed}')
        pretrained_path_prompt = os.path.join(checkpoint_path, 'P-MLM_Adapt', f'Seed-{domain_seed}')

        for fine_tune_repetition in range(args.repetitions):
            fine_tune_seed = args.seed[fine_tune_repetition]
            set_seed(fine_tune_seed)
            trainer = Trainer(args)
            info = f'Domain{domain_repetition}Seed{domain_seed}Finetune{fine_tune_repetition}Seed{fine_tune_seed}'
            logger.info(info)

            # logger.info("fine_tune start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune')
            # fine_tune_results.append(trainer.train(fine_tune_examples, eval_examples=eval_examples))

            logger.info("fine_tune_with_mlm_adapted start: ")
            args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_mlm_adapted')
            fine_tune_with_mlm_adapted_results.append(
                trainer.train(fine_tune_examples,
                              eval_examples=eval_examples,
                              checkpoint_path=pretrained_path))

            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_mlm_adapted')
            #
            # fine_tune_with_mlm_adapted_results.append(
            #     trainer.train(fine_tune_examples,
            #                   eval_examples=eval_examples,
            #                   # checkpoint_path=pretrained_path
            #                   ))

            # logger.info("fine_tune_with_prompt_mlm_adapted start: ")
            # args.saved_path = os.path.join(args.output_dir, info, 'fine_tune_with_prompt_mlm_adapted')
            # fine_tune_with_prompt_mlm_adapted_results.append(
            #     trainer.train(fine_tune_examples,
            #                   eval_examples=eval_examples,
            #                   checkpoint_path=pretrained_path_prompt))

    logger.info('fine_tune_results:')
    logger_helper(fine_tune_results, args.metrics)
    logger.info('fine_tune_with_mlm_adapted_results:')
    logger_helper(fine_tune_with_mlm_adapted_results, args.metrics)
    logger.info('fine_tune_with_prompt_mlm_adapted_results:')
    logger_helper(fine_tune_with_prompt_mlm_adapted_results, args.metrics)


def adapt_train(args):
    for domain_repetition in range(args.repetitions):
        domain_seed = args.seed[domain_repetition] + 100
        set_seed(domain_seed)
        unlabeled_examples, _ = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                                              num_examples=1.-args.train_examples, seed=domain_seed)

        # more_examples, _ = load_examples('qnli', './data/qnli', TRAIN_SET,
        #                                  num_examples=1. - args.train_examples, seed=domain_seed)
        #
        # pretrained_path = os.path.join(args.output_dir, 'MLM_Adapt', f'Seed-{domain_seed}')
        # AdaptTrainer('mlm_adapt', args.data_dir, pretrained_path, args.model_name_or_path,
        #              args.task_name, 128, args.device, args.n_gpu, args.pattern_id).train(
        #     unlabeled_examples + more_examples)

        pretrained_path = os.path.join('./output/sst-2/bert-base-uncased', 'MLM_Adapt', f'Seed-{domain_seed}')
        AdaptTrainer('mlm_adapt', args.data_dir, pretrained_path, args.model_name_or_path,
                     args.task_name, args.max_length, args.device, args.n_gpu, args.pattern_id).train(
            unlabeled_examples)

        # pretrained_path = os.path.join(args.output_dir, 'P-MLM_Adapt', f'Seed-{domain_seed}')
        # AdaptTrainer('prompt_mlm_adapt', args.data_dir, pretrained_path, args.model_name_or_path,
        #              args.task_name, args.max_length, args.device, args.n_gpu, args.pattern_id).train(
        #     unlabeled_examples)
