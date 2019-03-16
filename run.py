#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic parsing
run.py: Run Script for semantic parsing given a domain
Amir Ziai
Adapted from
    - CS224N NMT code
    - Jia et al [https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f/]
    - OpenNMT-py [https://github.com/OpenNMT/OpenNMT-py]

Usage:
    run.py train --file-path-train=<file> --file-path-dev=<file> [options]
    run.py decode --file-path-train=<file> --file-path-dev=<file> [options]
    run.py train decode --file-path-train=<file> --file-path-dev=<file> [options]

Options:
    -h --help                               show this screen.
    --domain-name=<str>                     dataset domain name [default: geoquery]
    --file-path-train=<file>                train file
    --file-path-dev=<file>                  dev file
    --seed=<int>                            seed [default: 0]
    --embed-size=<int>                      embedding size [default: 128]
    --hidden-size=<int>                     hidden size [default: 128]
    --dropout=<float>                       dropout [default: 0.3]
    --lr=<float>                            learning rate [default: 0.01]
    --cuda                                  use GPU
    --batch-size-train=<int>                batch size for training [default: 32]
    --batch-size-dev=<int>                  batch size for dev [default: 512]
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --max-epoch=<int>                       max epoch [default: 100]
    --file-path-model=<file>                model save path [default: model.bin]
    --beam-size=<int>                       beam size [default: 5]
    --augment=<str>                         augment type [default: None]
    --aug-frac=<float>                      augmentation fraction [default: 1.0]
    --max-sentence-length=<int>             max sentence len [default: 1000]
    --encoder-type=<str>                    encoder type [default: brnn]
    --decoder-type=<str>                    decoder type [default: rnn]
    --pre-train                             pre-train
    --results-dir=<str>                     results directory [default: results]
    --python-cmd=<str>                      python command to use [default: python3]
"""
import glob
import math
import os
import random
import subprocess
from collections import namedtuple
from itertools import repeat
from typing import List, Optional, Tuple, NamedTuple

import numpy as np
import torch
from docopt import docopt

import domains
from augmentation import Augmenter
from domains import DomainData
from onmt.translate.translator import build_translator
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from utils import log, unzip, hash_dict

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class ExperimentResults(NamedTuple):
    sequence_correct: int
    sequence_total: int
    sequence_accuracy: float
    token_correct: int
    token_total: int
    token_accuracy: float
    denotation_correct: Optional[int]
    denotation_total: Optional[int]
    denotation_accuracy: Optional[float]


class Runner:
    def __init__(self,
                 domain_name: str,
                 file_path_train: str,
                 file_path_dev: str,
                 seed: int,
                 embed_size: int,
                 hidden_size: int,
                 dropout: float,
                 lr: float,
                 cuda: bool,
                 batch_size_train: int,
                 batch_size_dev: int,
                 valid_niter: int,
                 max_epoch: int,
                 file_path_model: str,
                 beam_size: int,
                 augment: Optional[str],
                 aug_frac: float,
                 max_sentence_length: int,
                 encoder_type: str,
                 decoder_type: str,
                 pre_train: bool,
                 skip_training: bool = False,
                 results_dir: str = 'results',
                 python_cmd: str = 'python3'):
        # params
        self.domain_name = domain_name
        self.file_path_train = file_path_train
        self.file_path_dev = file_path_dev
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.cuda = cuda
        self.batch_size_train = batch_size_train
        self.batch_size_dev = batch_size_dev
        self.valid_niter = valid_niter
        self.max_epoch = max_epoch
        self.file_path_model = file_path_model
        self.beam_size = beam_size
        self.seed = seed
        self.augment = augment
        self.aug_frac = aug_frac
        self.max_sentence_length = max_sentence_length
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.pre_train = pre_train
        self.skip_training = skip_training
        self.results_dir = results_dir
        self.python_cmd = python_cmd

        self.uuid = self._get_uuid()

        # seed the random number generators
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # important things to populate later
        self.domain = None
        self.data_train = None
        self.data_dev = None
        self.vocab = None
        self.model = None
        self.optimizer = None
        self.augmenter = None

    def _get_uuid(self):
        return hash_dict(self.__dict__)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _domain_data_to_raw(data: DomainData) -> List[Tuple[str, str]]:
        return [(' '.join(x_lst), ' '.join(y_lst)) for x_lst, y_lst in data]

    @staticmethod
    def _raw_to_domain_data(data: List[Tuple[str, str]]) -> DomainData:
        return [(x.split(), y.split()) for x, y in data]

    @staticmethod
    def _load_data(file_path: str, domain: domains.Domain) -> DomainData:
        dataset = []
        with open(file_path) as f:
            for line in f:
                x_str, y_str = line.rstrip('\n').split('\t')
                if domain:
                    y_str = domain.preprocess_lf(y_str)

                x = x_str.split()
                y = y_str.split()
                dataset.append((x, y))

        return dataset

    @staticmethod
    def _get_accuracy_metrics(is_correct_list: List[int], tokens_correct_list: List[int], y_len_list: List[int],
                              denotation_correct_list: List) -> ExperimentResults:
        # Overall metrics
        num_examples = len(is_correct_list)
        num_correct = sum(is_correct_list)
        num_tokens_correct = sum(tokens_correct_list)
        num_tokens = sum(y_len_list)
        seq_accuracy = float(num_correct) / num_examples
        token_accuracy = float(num_tokens_correct) / num_tokens

        # sequence-level accuracy
        log('Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy))
        log('Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy))

        # denotation-level accuracy
        denotation_correct = denotation_accuracy = denotation_total = None
        if denotation_correct_list:
            denotation_correct = sum(denotation_correct_list)
            denotation_total = num_examples
            denotation_accuracy = float(denotation_correct) / denotation_total
            log('Denotation-level accuracy: %d/%d = %g' % (denotation_correct, num_examples, denotation_accuracy))

        return ExperimentResults(num_correct, num_examples, seq_accuracy,
                                 num_tokens_correct, num_tokens, token_accuracy,
                                 denotation_correct, denotation_total, denotation_accuracy)

    def _evaluate(self, data: DomainData, hypotheses: List[List[Hypothesis]]) -> ExperimentResults:
        xs, ys = unzip(data)
        true_answers = [''.join(y) for y in ys]

        derivs, denotation_correct_list = self.domain.compare_answers(true_answers, hypotheses)

        is_correct_list = []
        tokens_correct_list = []
        y_len_list = []

        for x, y, y_str, deriv in zip(xs, ys, true_answers, derivs):
            y_pred_toks = deriv.value
            y_pred_str = ''.join(y_pred_toks)

            # Compute accuracy metrics
            is_correct = (y_pred_str == y_str)
            tokens_correct = sum(a == b for a, b in zip(y_pred_toks, y))
            is_correct_list.append(is_correct)
            tokens_correct_list.append(tokens_correct)
            y_len_list.append(len(y))

        return self._get_accuracy_metrics(is_correct_list, tokens_correct_list, y_len_list, denotation_correct_list)

    def _get_opennmt_file_name(self, file_path: str, src: bool) -> str:
        file_name = file_path.split('/')[-1]
        name, _ = file_name.split('.')
        return f"{self.results_dir}/{self.uuid}_{name}_{'src' if src else 'tgt'}"

    def _write_to_file(self, sentences: List[List[str]], train: bool, src: bool) -> None:
        file_name_orig = self.file_path_train if train else self.file_path_dev
        file_name = self._get_opennmt_file_name(file_name_orig, src)

        with open(file_name, 'w') as f:
            for sent in sentences:
                f.write(f"{' '.join(sent).strip()}\n")

    @property
    def opennmt_files_prefix(self) -> str:
        return f'{self.results_dir}/{self.uuid}_{self.domain_name}'

    def _remove_opennmt_files(self) -> None:
        for fl in glob.glob(f'{self.opennmt_files_prefix}*.pt'):
            os.remove(fl)

    def prep_data(self, use_augmentation: bool) -> str:
        self.domain = domains.new(self.domain_name)

        self._remove_opennmt_files()

        self.data_train = self._load_data(self.file_path_train, self.domain)
        log(f'Train data size: {len(self.data_train)}')

        # augmenter
        if use_augmentation and self.augment:
            aug_types = self.augment.split('+')
            data_train_raw = self._domain_data_to_raw(self.data_train)
            self.augmenter = Augmenter(self.domain, data_train_raw, aug_types)
            # Do data augmentation on the fly
            aug_num = int(round(self.aug_frac * len(self.data_train)))
            aug_exs = self.augmenter.sample(aug_num)
            self.data_train += self._raw_to_domain_data(aug_exs)
            random.shuffle(self.data_train)

        log(f'Augmented train data size: {len(self.data_train)}')
        self.data_dev = self._load_data(self.file_path_dev, self.domain)
        log(f'Dev train data size: {len(self.data_dev)}')
        train_src, train_tgt = unzip(self.data_train)
        dev_src, dev_tgt = unzip(self.data_dev)
        self._write_to_file(train_src, train=True, src=True)
        self._write_to_file(train_tgt, train=True, src=False)
        self._write_to_file(dev_src, train=False, src=True)
        self._write_to_file(dev_tgt, train=False, src=False)

        # pre-process
        cmd = [
            'python3', 'preprocess.py',
            '-save_data', self.opennmt_files_prefix,
            '-train_src', self._get_opennmt_file_name(self.file_path_train, src=True),
            '-train_tgt', self._get_opennmt_file_name(self.file_path_train, src=False),
            '-valid_src', self._get_opennmt_file_name(self.file_path_dev, src=True),
            '-valid_tgt', self._get_opennmt_file_name(self.file_path_dev, src=False),
            '--src_seq_length', str(self.max_sentence_length),
            '--tgt_seq_length', str(self.max_sentence_length),
            '--log_file', self._get_process_log_file_name('pre_process'),
            '--dynamic_dict'
        ]
        return self._subprocess_runner(cmd)

    @property
    def file_path_model_uuid(self) -> str:
        return f'{self.results_dir}/{self.uuid}_{self.file_path_model}'

    @property
    def file_path_model_uuid_checkpoint(self) -> str:
        return f'{self.file_path_model_uuid}_step_{self.max_epoch}.pt'

    def train(self, fine_tune: bool) -> str:
        cmd = [
            'python3', 'train.py',
            '-data', self.opennmt_files_prefix,
            '-save_model', self.file_path_model_uuid,
            '--copy_attn',
            '--train_steps', str(self.max_epoch),
            '--optim', 'adam',
            '--learning_rate', str(self.lr),
            '--valid_steps', str(self.valid_niter),
            '--word_vec_size', str(self.embed_size),
            '--layers', '1',
            '--rnn_size', str(self.hidden_size),
            '--dropout', str(self.dropout),
            '--batch_size', str(self.batch_size_train),
            '--seed', str(self.seed),
            '--encoder_type', self.encoder_type,
            '--log_file', self._get_process_log_file_name(f"train{'_fine_tune' if fine_tune else ''}"),
            '--decoder_type', self.decoder_type
        ]
        cmd += (['-copy_attn_force'] if self.decoder_type != 'transformer' else [])
        cmd += (['--train_from', self.file_path_model_uuid_checkpoint] if fine_tune else [])

        return self._subprocess_runner(cmd)

    def _get_process_log_file_name(self, name: str) -> str:
        return f'{self.results_dir}/{self.uuid}_log_{name}'

    @staticmethod
    def _subprocess_runner(cmd: List[str]) -> str:
        log(' '.join(cmd))
        output_bytes = subprocess.check_output(cmd)
        return output_bytes.decode('utf-8')

    @staticmethod
    def _dict_to_namedtuple(d: dict) -> namedtuple:
        Temp = namedtuple('Temp', sorted(d))
        return Temp(**d)

    @property
    def file_path_translate_output(self):
        return f'{self.results_dir}/{self.uuid}_final_output'

    def decode(self) -> ExperimentResults:
        args = dict(alpha=0.0,
                    attn_debug=False,  # TODO: this is cool, do something with it
                    avg_raw_probs=False,
                    batch_size=self.batch_size_dev,
                    beam_size=self.beam_size,
                    beta=-0.0,
                    block_ngram_repeat=0,
                    config=None,
                    coverage_penalty='none',
                    data_type='text',
                    dump_beam='',
                    dynamic_dict=True,
                    fp32=False,
                    gpu=-1,
                    ignore_when_blocking=[],
                    image_channel_size=3,
                    length_penalty='none',
                    log_file='',
                    log_file_level='0',
                    max_length=100,
                    max_sent_length=None,
                    min_length=0,
                    models=[self.file_path_model_uuid_checkpoint],
                    n_best=self.beam_size,
                    output=self.file_path_translate_output,
                    random_sampling_temp=1.0,
                    random_sampling_topk=1,
                    ratio=-0.0,
                    replace_unk=True,
                    report_bleu=False,
                    report_rouge=False,
                    report_time=True,
                    sample_rate=16000,
                    save_config=None,
                    seed=self.seed,
                    shard_size=10000,
                    share_vocab=False,
                    src=self._get_opennmt_file_name(self.file_path_dev, src=True),
                    tgt=self._get_opennmt_file_name(self.file_path_dev, src=False),
                    src_dir='',
                    stepwise_penalty=False,
                    verbose=True, window='hamming', window_size=0.02, window_stride=0.01)

        opt = self._dict_to_namedtuple(args)

        logger = init_logger(opt.log_file)
        translator = build_translator(opt, report_score=True)
        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, opt.shard_size) if opt.tgt is not None else repeat(None)
        shard_pairs = zip(src_shards, tgt_shards)

        xs = []
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            xs += translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                attn_debug=opt.attn_debug
            )

        all_scores, all_predictions = xs
        hypotheses = [
            [
                Hypothesis(value=prediction.split(), score=math.exp(score))
                for score, prediction in zip(scores, predictions)
            ]
            for scores, predictions in zip(all_scores, all_predictions)
        ]

        return self._evaluate(self.data_dev, hypotheses)

    def run(self) -> ExperimentResults:
        self.prep_data(use_augmentation=self.augment is not None)

        if not self.skip_training:
            if self.augment and self.pre_train:
                self.train(fine_tune=False)
                self.prep_data(use_augmentation=False)
                self.train(fine_tune=True)
            else:
                self.train(fine_tune=False)

        return self.decode()


def main() -> None:
    """
    Parse args and run.
    """
    args = docopt(__doc__)

    # Check PyTorch version
    assert torch.__version__ == "1.0.0", f"You have PyTorch=={torch.__version__} and you should have version 1.0.0"
    augment = args['--augment']

    runner = Runner(
        domain_name=args['--domain-name'],
        file_path_train=args['--file-path-train'],
        file_path_dev=args['--file-path-dev'],
        seed=int(args['--seed']),
        embed_size=int(args['--embed-size']),
        hidden_size=int(args['--hidden-size']),
        dropout=float(args['--dropout']),
        lr=float(args['--lr']),
        cuda=args['--cuda'],
        batch_size_train=int(args['--batch-size-train']),
        batch_size_dev=int(args['--batch-size-dev']),
        valid_niter=int(args['--valid-niter']),
        max_epoch=int(args['--max-epoch']),
        file_path_model=args['--file-path-model'],
        beam_size=int(args['--beam-size']),
        augment=augment if augment != 'None' else None,
        aug_frac=float(args['--aug-frac']),
        max_sentence_length=int(args['--max-sentence-length']),
        encoder_type=args['--encoder-type'],
        decoder_type=args['--decoder-type'],
        pre_train=args['--pre-train'],
        skip_training=not args['train'],
        python_cmd=args['--python-cmd']
    )

    runner.run()


if __name__ == '__main__':
    main()
