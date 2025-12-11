#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import time
from pathlib import Path

import sacrebleu
from model import Hypothesis, HieroLM
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils

from parse import args
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter

import time

def evaluate_ppl(model, dev_data, batch_size, device):
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents, device).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def evaluate_accuracy_and_f1(model, dev_data, batch_size, device):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_base = 0

        preds = []
        truths = []

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            predictions, target_masks, target_padded, source_lenths = model.predict(src_sents, tgt_sents, device)

            correct_num = ((predictions == target_padded) * target_masks).sum()
            base_num = target_masks.sum()

            total_correct += correct_num
            total_base += base_num

            for i in range(predictions.shape[1]):
                preds.append(predictions[:,i][:source_lenths[i]])
                truths.append(target_padded[:,i][:source_lenths[i]])
        
        preds = torch.cat(preds).cpu().numpy()
        truths = torch.cat(truths).cpu().numpy()
        f1 = f1_score(truths, preds, average="macro")

        accuracy = total_correct/total_base

    if was_training:
        model.train()

    return accuracy, f1


def train(args):

    #### LOAD DATA

    train_file = "data/"+args.dataset+"/"+args.train_file
    train_data_src, train_data_tgt = read_corpus(train_file)

    print("loaded training set from", train_file)

    dev_file = "data/"+args.dataset+"/"+args.dev_file
    dev_data_src, dev_data_tgt = read_corpus(dev_file)

    print("loaded dev set from", dev_file)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = args.train_batch_size
    clip_grad = args.clip_grad
    valid_niter = args.valid_niter
    log_every = args.log_every
    model_save_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_save_path

    vocab_file = "data/"+args.dataset+"/"+args.vocab_file
    vocab = Vocab.load(vocab_file)

    #### INITIALIZE MODEL

    model = HieroLM(embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab)

    model.train()

    tensorboard_path = "lstm" if args.cuda else "lstm_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

    #### INITIALIZE MODEL PARAMS

    uniform_init = args.uniform_init
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    #### VOCAB MASKS

    vocab_mask = torch.ones(len(vocab.vocab))
    vocab_mask[vocab.vocab['<pad>']] = 0

    #### PREPARE TRAINING

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    max_epoch = args.max_epoch

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            #### MODEL INPUT 

            example_losses = -model(src_sents, tgt_sents, device) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                writer.add_scalar("loss/train", report_loss / report_tgt_words, train_iter)
                writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), train_iter)
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_tgt_words,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                writer.add_scalar("loss/val", cum_loss / cum_tgt_words, train_iter)
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_tgt_words,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl, accuracy, and f1
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128, device=device)
                dev_accuracy, dev_f1 = evaluate_accuracy_and_f1(model, dev_data, batch_size=128, device=device)
                
                valid_metric = -dev_ppl
                writer.add_scalar("perplexity/val", dev_ppl, train_iter)

                print('validation: iter %d, dev. ppl %f, dev. accuracy %f, dev. f1 %f' % (train_iter, dev_ppl, dev_accuracy, dev_f1), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')

                elif patience < int(args.patience):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args.patience):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args.max_num_trial):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage, weights_only=False)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == max_epoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def decode(args: Dict[str, str]):
    test_file = "data/"+args.dataset+"/"+args.test_file
    print("load test source sentences from [{}]".format(test_file), file=sys.stderr)
    test_data_src, test_data_tgt = read_corpus(test_file)

    test_data = list(zip(test_data_src, test_data_tgt))

    model_load_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_path

    print("load model from {}".format(model_load_path), file=sys.stderr)
    model = HieroLM.load(model_load_path)

    if args.cuda:
        device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)

    test_ppl = evaluate_ppl(model, test_data, batch_size=128, device=device)
    test_accuracy, test_f1 = evaluate_accuracy_and_f1(model, test_data, batch_size=128, device=device)

    print('test: ppl %f, accuracy %f, f1 %f' % (test_ppl, test_accuracy, test_f1), file=sys.stderr)

def realtime(args):
    model_load_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_path
    print("load model from {}".format(model_load_path), file=sys.stderr)
    model = HieroLM.load(model_load_path)
    if args.cuda:
        device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    with torch.no_grad():
        while True:
            src = [input("Input:").split(" ")]
            prediction = model.predict_realtime(src, device)
            print("Next word:", prediction)



def main():

    if args.mode=="train":
        train(args)
    elif args.mode=="decode":
        decode(args)
    elif args.mode=="realtime":
        realtime(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
