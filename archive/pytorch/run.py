#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic parsing
run.py: Run Script for semantic parsing given a domain
Amir Ziai <amirziai@stanford.edu>
Adapted from CS224N NMT assignment code

Usage:
    run.py train --file-path-train=<file> --file-path-dev=<file> [options]
    run.py decode --file-path-train=<file> --file-path-dev=<file> [options]
    run.py train decode --file-path-train=<file> --file-path-dev=<file> [options]

Options:
    -h --help                               show this screen.
    --domain-name=<str>                     dataset domain name [default: geoquery]
    --file-path-train=<file>                train file
    --file-path-dev=<file>                  dev file
    --vocab-size=<int>                      vocab size [default: 10000]
    --seed=<int>                            seed [default: 0]
    --freq-cutoff=<int>                     tokens less freuqent than this are cut out from vocab [default: 1]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --dropout=<float>                       dropout [default: 0.3]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --lr=<float>                            learning rate [default: 0.001]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --cuda                                  use GPU
    --batch-size-train=<int>                batch size for training [default: 32]
    --batch-size-dev=<int>                  batch size for dev [default: 128]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --file-path-model=<file>                model save path [default: model.bin]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --beam-size=<int>                       beam size [default: 5]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import time
from typing import List

import numpy as np
import torch
from docopt import docopt
from tqdm import tqdm

import domains
from domains import DomainData
from nmt_model import NMT, Hypothesis
from utils import log, batch_iter, unzip
from vocab import Vocab


class Runner:
    def __init__(self,
                 domain_name: str,
                 file_path_train: str,
                 file_path_dev: str,
                 vocab_size: int,
                 seed: int,
                 freq_cutoff: int,
                 embed_size: int,
                 hidden_size: int,
                 dropout: float,
                 uniform_init: float,
                 lr: float,
                 lr_decay: float,
                 cuda: bool,
                 batch_size_train: int,
                 batch_size_dev: int,
                 clip_grad: float,
                 log_every: int,
                 valid_niter: int,
                 max_epoch: int,
                 patience: int,
                 file_path_model: str,
                 max_num_trial: int,
                 beam_size: int,
                 max_decoding_time_step: int):
        # params
        self.domain_name = domain_name
        self.file_path_train = file_path_train
        self.file_path_dev = file_path_dev
        self.vocab_size = vocab_size
        self.freq_cutoff = freq_cutoff
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.uniform_init = uniform_init
        self.lr = lr
        self.lr_decay = lr_decay
        self.cuda = cuda
        self.batch_size_train = batch_size_train
        self.batch_size_dev = batch_size_dev
        self.clip_grad = clip_grad
        self.log_every = log_every
        self.valid_niter = valid_niter
        self.max_epoch = max_epoch
        self.patience = patience
        self.file_path_model = file_path_model
        self.max_num_trial = max_num_trial
        self.beam_size = beam_size
        self.max_decoding_time_step = max_decoding_time_step
        self.seed = seed

        # seed the random number generators
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        # important things to populate later
        self.domain = None
        self.data_train = None
        self.data_dev = None
        self.vocab = None
        self.model = None
        self.device = None
        self.optimizer = None

    @staticmethod
    def _load_data(file_path: str, domain: domains.Domain) -> DomainData:
        dataset = []
        with open(file_path) as f:
            for line in f:
                x_str, y_str = line.rstrip('\n').split('\t')
                if domain:
                    y_str = domain.preprocess_lf(y_str)

                x = x_str.split()
                # TODO: </s> seems to already exist somewhere
                # y = ['<s>'] + y_str.split() + ['</s>']
                y = ['<s>'] + y_str.split()  # + ['</s>']
                dataset.append((x, y))

        return dataset

    def _init_params(self):
        if np.abs(self.uniform_init) > 0.:
            log('uniformly initialize parameters [-%f, +%f]' % (self.uniform_init, self.uniform_init))
            for p in self.model.parameters():
                p.data.uniform_(-self.uniform_init, self.uniform_init)

    def _vocab_mask(self) -> None:
        vocab_mask = torch.ones(len(self.vocab.tgt))
        vocab_mask[self.vocab.tgt['<pad>']] = 0

    def _init_device(self) -> None:
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        log('use device: %s' % self.device)
        self.model = self.model.to(self.device)

    def _init_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _evaluate_perplexity(self) -> float:
        was_training = self.model.training
        self.model.eval()

        cum_loss = 0.
        cum_tgt_words = 0.

        # no_grad() signals backend to throw away all gradients
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(self.data_dev, self.batch_size_dev):
                loss = -self.model(src_sents, tgt_sents).sum()

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        if was_training:
            self.model.train()

        return ppl

    def _train_loop(self):
        num_trial = 0
        train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
        cum_examples = report_examples = epoch = valid_num = 0
        hist_valid_scores = []
        train_time = begin_time = time.time()
        log('begin Maximum Likelihood training')

        log(f'Train dataset has {len(self.data_train)} items')
        early_stop = False

        while epoch < self.max_epoch and not early_stop:
            epoch += 1

            # TODO: add augmenter here to extend data
            # TODO: add concatenated examples

            for src_sents, tgt_sents in batch_iter(self.data_train, batch_size=self.batch_size_train, shuffle=True):
                train_iter += 1

                self.optimizer.zero_grad()

                batch_size = len(src_sents)

                example_losses = -self.model(src_sents, tgt_sents)  # (batch_size,)
                batch_loss = example_losses.sum()
                loss = batch_loss / batch_size

                loss.backward()

                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                self.optimizer.step()

                batch_losses_val = batch_loss.item()
                report_loss += batch_losses_val
                cum_loss += batch_losses_val

                # TODO: why remove <s>?
                tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                report_tgt_words += tgt_words_num_to_predict
                cum_tgt_words += tgt_words_num_to_predict
                report_examples += batch_size
                cum_examples += batch_size

                if train_iter % self.log_every == 0:
                    log('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                        'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                           report_loss /
                                                                                           report_examples,
                                                                                           math.exp(
                                                                                               report_loss /
                                                                                               report_tgt_words),
                                                                                           cum_examples,
                                                                                           report_tgt_words / (
                                                                                                   time.time() -
                                                                                                   train_time),
                                                                                           time.time() - begin_time))
                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.

                # perform validation
                if train_iter % self.valid_niter == 0:
                    log('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch,
                                                                                               train_iter,
                                                                                               cum_loss / cum_examples,
                                                                                               np.exp(cum_loss /
                                                                                                      cum_tgt_words),
                                                                                               cum_examples))

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)

                    # dev batch size can be a bit larger
                    dev_ppl = self._evaluate_perplexity()
                    valid_metric = -dev_ppl

                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)

                    if is_better:
                        patience = 0
                        log('save currently the best model to [%s]' % self.file_path_model)
                        self.model.save(self.file_path_model)

                        # also save the optimizers' state
                        torch.save(self.optimizer.state_dict(), self.file_path_model + '.optim')
                    elif patience < self.patience:
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == self.patience:
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == self.max_num_trial:
                                log('early stop!')
                                early_stop = True
                                break

                            # decay lr, and restore from previously best checkpoint
                            lr = self.optimizer.param_groups[0]['lr'] * float(self.lr_decay)
                            log('load previously best model and decay learning rate to %f' % lr)

                            # load model
                            params = torch.load(self.file_path_model, map_location=lambda storage, loc: storage)
                            self.model.load_state_dict(params['state_dict'])
                            self.model = self.model.to(self.device)

                            log('restore parameters of the optimizers')
                            self.optimizer.load_state_dict(torch.load(self.file_path_model + '.optim'))

                            # set new lr
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0

        if not early_stop and epoch >= self.max_epoch:
            log(f'reached epoch {epoch} which is >= {self.max_epoch}  maximum number of epochs!')

    def train(self) -> None:
        log('training')

        # get data by domain
        self.domain = domains.new(self.domain_name)
        self.data_train = self._load_data(self.file_path_train, self.domain)
        self.data_dev = self._load_data(self.file_path_dev, self.domain)
        train_src, train_tgt = unzip(self.data_train)

        # build the vocabulary from the training data
        self.vocab = Vocab.build(train_src, train_tgt, vocab_size=self.vocab_size, freq_cutoff=self.freq_cutoff)

        # build the model with the provided params
        self.model = NMT(embed_size=self.embed_size,
                         hidden_size=self.hidden_size, dropout_rate=self.dropout, vocab=self.vocab)

        # model architecture and parameters
        log(str(self.model.train()))

        # init
        self._init_params()
        self._vocab_mask()
        self._init_optimizer()

        # train
        self._train_loop()

        log('training done!')

    def _load_model(self) -> None:
        if not self.model:
            log("load model from {}".format(self.file_path_model))
            self.model = NMT.load(self.file_path_model)
            self._init_device()

    @staticmethod
    def _log_accuracy_metrics(is_correct_list: List[int], tokens_correct_list: List[int], y_len_list: List[int],
                              denotation_correct_list: List) -> None:
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
        if denotation_correct_list:
            denotation_correct = sum(denotation_correct_list)
            denotation_accuracy = float(denotation_correct) / num_examples
            log('Denotation-level accuracy: %d/%d = %g' % (denotation_correct, num_examples, denotation_accuracy))

    def _evaluate(self, data: DomainData, hypotheses: List[List[Hypothesis]]) -> None:
        xs, ys = unzip(data)
        true_answers = [' '.join(y) for y in ys]

        derivs, denotation_correct_list = self.domain.compare_answers(true_answers, hypotheses)

        is_correct_list = []
        tokens_correct_list = []
        y_len_list = []

        for x, y, y_str, deriv in zip(xs, ys, true_answers, derivs):
            y_pred_toks = deriv.value
            y_pred_str = ' '.join(y_pred_toks)

            # Compute accuracy metrics
            is_correct = (y_pred_str == y_str)
            tokens_correct = sum(a == b for a, b in zip(y_pred_toks, y))
            is_correct_list.append(is_correct)
            tokens_correct_list.append(tokens_correct)
            y_len_list.append(len(y))

        self._log_accuracy_metrics(is_correct_list, tokens_correct_list, y_len_list, denotation_correct_list)

    def decode(self) -> None:
        # data
        log("load test source sentences from [{}]".format(self.file_path_dev))
        if not self.data_dev:
            self.data_dev = self._load_data(self.file_path_dev, self.domain)
        if not self.domain:
            self.domain = domains.new(self.domain_name)

        dev_src, dev_tgt = unzip(self.data_dev)

        # model
        self._load_model()

        hypotheses = self.beam_search(dev_src)
        self._evaluate(self.data_dev, hypotheses)

    def beam_search(self, data_src: List[List[str]]) -> List[List[Hypothesis]]:
        """
        Run beam search to construct hypotheses for a list of natural language utterances.
        @param data_src: List of list of tokens.
        @returns hypotheses: List of Hypothesis logical forms.
        """
        was_training = self.model.training
        self.model.eval()

        hypotheses = []
        with torch.no_grad():
            for src_sent in tqdm(data_src, desc='Decoding', file=sys.stdout):
                example_hyps = self.model.beam_search(src_sent, beam_size=self.beam_size,
                                                      max_decoding_time_step=self.max_decoding_time_step)

                hypotheses.append(example_hyps)

        if was_training:
            self.model.train(was_training)

        return hypotheses


def main() -> None:
    """
    Parse args and run.
    """
    args = docopt(__doc__)

    # Check PyTorch version
    assert torch.__version__ == "1.0.0", f"You have PyTorch=={torch.__version__} and you should have version 1.0.0"

    runner = Runner(
        domain_name=args['--domain-name'],
        file_path_train=args['--file-path-train'],
        file_path_dev=args['--file-path-dev'],
        vocab_size=int(args['--vocab-size']),
        seed=int(args['--seed']),
        freq_cutoff=int(args['--freq-cutoff']),
        embed_size=int(args['--embed-size']),
        hidden_size=int(args['--hidden-size']),
        dropout=float(args['--dropout']),
        uniform_init=float(args['--uniform-init']),
        lr=float(args['--lr']),
        lr_decay=float(args['--lr-decay']),
        cuda=args['--cuda'],
        batch_size_train=int(args['--batch-size-train']),
        batch_size_dev=int(args['--batch-size-dev']),
        clip_grad=float(args['--clip-grad']),
        log_every=int(args['--log-every']),
        valid_niter=int(args['--valid-niter']),
        max_epoch=int(args['--max-epoch']),
        patience=int(args['--patience']),
        file_path_model=args['--file-path-model'],
        max_num_trial=int(args['--max-num-trial']),
        beam_size=int(args['--beam-size']),
        max_decoding_time_step=int(args['--max-decoding-time-step'])
    )

    if args['train']:
        runner.train()

    if args['decode']:
        runner.decode()


if __name__ == '__main__':
    main()
