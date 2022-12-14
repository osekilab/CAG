#!/usr/bin/env python3

# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/train.py

import argparse
import gc
import itertools
import logging
import os
import time

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from models import FixedStackCompositionAttentionGrammar
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from utils import pad_items
from warmup_scheduler import GradualWarmupScheduler

from data import Dataset, DatasetForAccelerator, GPT2Vocabulary, SentencePieceVocabulary
from transformers import AdamW

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-no-unk-train.json')
parser.add_argument('--val_file', default='data/ptb-no-unk-val.json')
parser.add_argument('--train_from', default='')
parser.add_argument(
    '--sp_model',
    default='',
    help='Subword-tokenized treebank should be trained with this argument. Path to trained sentencepiece model.',
)
parser.add_argument(
    '--tokenizer_model',
    default='',
    help='Subword-tokenized treebank should be trained with this argument. Path to trained GPT2Tokenizer().',
)
# Model options
parser.add_argument('--fixed_stack', action='store_true')
parser.add_argument('--strategy', default='top_down', choices=['top_down', 'in_order'])
parser.add_argument(
    '--w_dim', default=256, type=int, help='input/output word dimension'
)
parser.add_argument(
    '--h_dim', default=256, type=int, help='Transformer hidden dimension'
)
parser.add_argument(
    '--num_layers',
    default=2,
    type=int,
    help='number of layers in LM and the stack Transformer (for CompositionAttentionGrammar)',
)
parser.add_argument(
    '--num_heads',
    default=2,
    type=int,
    help='number of heads in LM and the stack Transformer (for CompositionAttentionGrammar)',
)
parser.add_argument(
    '--max_stack_size',
    default=100,
    type=int,
    help='The maximum stack size that this model might ever be used with.',
)
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--embd_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument(
    '--composition',
    default='lstm',
    choices=['lstm', 'attention'],
    help='lstm: original lstm composition; attention: gated attention introduced in Kuncoro et al. (2017).',
)
parser.add_argument(
    '--not_swap_in_order_stack',
    action='store_true',
    help=(
        'If True, prevent swapping elements by an open action for the in-order system.'
        'WARNING: when --fixed_stack is True,\
        this option is automatically and always set to True (obsolete option and no need to care)'
    ),
)
# Optimization options
parser.add_argument(
    '--batch_group',
    choices=['same_length', 'random', 'similar_length', 'similar_action_length'],
    default='similar_length',
    help='Sentences are grouped by this criterion to make each batch.',
)
parser.add_argument(
    '--max_group_length_diff',
    default=20,
    type=int,
    help='When --batch_group=similar_length or similar_action_length,\
        maximum (token or action) length difference in a single batch does not exceed this.',
)
parser.add_argument(
    '--group_sentence_size',
    default=1024,
    type=int,
    help='When --batch_group=similar_length, \
    sentences are first sorted by length and grouped by this number of sentences, \
    from which each batch is sampled.',
)
parser.add_argument(
    '--optimizer',
    default='adam',
    choices=['sgd', 'adam', 'adamW'],
    help='Which optimizer to use.',
)
parser.add_argument('--lr_scheduler', default=None, choices=['plateau', 'warmup'])
parser.add_argument(
    '--plateau_lr_decay',
    default=0.5,
    type=float,
    help='lr is decayed at this rate if --lr_scheduler=plateau',
)
parser.add_argument(
    '--plateau_lr_patience',
    default=1,
    type=int,
    help='Number of epochs with no improvement after which learning rate will be reduced\
    (patience for ReduceLROnPlateau).',
)
parser.add_argument(
    '--warmup_steps', default=10000, type=float, help='Total steps (batches) for warmup'
)
parser.add_argument(
    '--random_unk',
    action='store_true',
    help='Randomly replace a token to <unk> on training sentences \
    by a probability inversely proportional to word frequency.',
)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument(
    '--batch_token_size',
    type=int,
    default=15000,
    help='Number of tokens in a batch (batch_size*sentence_length) does not exceed this.',
)
parser.add_argument(
    '--batch_action_size',
    type=int,
    default=45000,
    help='(batch_size*max_action_length) does not exceed this.',
)
parser.add_argument(
    '--save_path', default='cag.pt', help='where to save the best model'
)
parser.add_argument(
    '--num_epochs', default=18, type=int, help='number of training epochs'
)
parser.add_argument(
    '--min_epochs',
    default=8,
    type=int,
    help='do not decay learning rate for at least this many epochs',
)
# parser.add_argument(
#    '--decay_cond_epochs',
#    default=1,
#    type=int,
#    help='decay learning rate if loss does not improve consecutively this many steps',
# )
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument(
    '--loss_normalize', default='batch', choices=['sum', 'batch', 'action', 'token']
)
# parser.add_argument('--decay', default=0.5, type=float, help='')
parser.add_argument(
    '--param_init',
    default=0,
    type=float,
    help='parameter initialization (over uniform)',
)
parser.add_argument(
    '--max_grad_norm', default=5, type=float, help='gradient clipping parameter'
)
# parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
# parser.add_argument(
#     '--device',
#     default='cuda',
#     choices=['cuda', 'cpu', 'parallel'],
#     help='If "cuda", GPU number --gpu is used.',
# )
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument(
    '--print_every', type=int, default=500, help='print stats after this many batches'
)
parser.add_argument(
    '--valid_every',
    type=int,
    default=-1,
    help='If > 0, validate and save model every this many batches',
)
parser.add_argument(
    '--tensorboard_log_dir',
    default='',
    help='If not empty, tensor board summaries are recorded on the directory `tensor_board_log_dir/save_path`',
)
parser.add_argument('--amp', action='store_true')
parser.add_argument(
    '--early_stop',
    action='store_true',
    help='Stop learning if loss monotonically increases --early_stop_patience times (default=5)',
)
parser.add_argument('--early_stop_patience', type=int, default=5)


class TensorBoardLogger(object):
    def __init__(self, args):
        if len(args.tensorboard_log_dir) > 0:
            log_dir = os.path.join(args.tensorboard_log_dir, args.save_path)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.global_step = 0
            self.start_time = time.time()
        else:
            self.writer = None

    def write(self, kvs={}, step=None, use_time=False):
        if self.writer is not None:
            if use_time:
                step = time.time() - self.start_time
            else:
                if step is None:
                    step = self.global_step
                    self.global_step += 1
            for k, v in kvs.items():
                self.writer.add_scalar(k, v, global_step=step)


def to_namespace(args):
    if isinstance(args, dict):
        # Args is saved as a dict so we need to convert to Namespace when loading from a checkpoint.
        from argparse import Namespace

        args = Namespace(**args)
    return args


def create_optimizer(args, model):
    args = to_namespace(args)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        args.min_epochs = args.num_epochs
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        args.min_epochs = args.num_epochs
    return optimizer


def create_scheduler(args, optimizer):
    args = to_namespace(args)
    if args.lr_scheduler == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            factor=args.plateau_lr_decay,
            patience=args.plateau_lr_patience,
            verbose=True,
        )
    elif args.lr_scheduler == 'warmup':
        return GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=args.warmup_steps
        )
    else:
        return StepLR(optimizer, 1000000000, 1.0)  # this scheduler does nothing.


def create_model(args, action_dict, vocab):
    args = to_namespace(args)
    model_args = {
        'action_dict': action_dict,
        'vocab_size': vocab.size(),
        'padding_idx': vocab.padding_idx,
        'word_dim': args.w_dim,
        'hidden_dim': args.h_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'max_stack_size': args.max_stack_size,
        'dropout': args.dropout,
        'embd_dropout': args.embd_dropout,
        'attention_composition': args.composition == 'attention',
    }

    if args.strategy == 'top_down':
        model = FixedStackCompositionAttentionGrammar(**model_args)
    # elif args.strategy == 'in_order':
    # model = FixedStackInOrderCompositionAttentionGrammar(**model_args)
    if args.param_init > 0:
        for param in model.parameters():
            param.data.uniform_(-args.param_init, args.param_init)
    return model


def main(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(fp16=args.amp, kwargs_handlers=kwargs_handlers)
    device = accelerator.device
    if accelerator.is_local_main_process:
        logger.info('Args: {}'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.device == 'cuda' or args.device == 'parallel':
    #     torch.cuda.manual_seed(args.seed)

    if len(args.sp_model) > 0:
        if accelerator.is_local_main_process:
            logger.info('Load sentencepiece vocabulary from {}'.format(args.sp_model))
        vocab = SentencePieceVocabulary(args.sp_model)
    elif len(args.tokenizer_model) > 0:
        if accelerator.is_local_main_process:
            logger.info(
                'Load tokenizer vocabulary from {}'.format(args.tokenizer_model)
            )
        vocab = GPT2Vocabulary(args.tokenizer_model)
    else:
        vocab = None

    def collate_fn(batch):
        token_ids, action_ids, subword_end_mask, max_stack_size = list(zip(*batch))

        token_ids = list(token_ids)
        action_ids = list(action_ids)
        subword_end_mask = list(subword_end_mask)
        max_stack_size = max(list(max_stack_size))

        action_ids = torch.tensor(
            pad_items(action_ids, train_data.action_dict.padding_idx)[0],
            dtype=torch.long,
        )
        token_ids = torch.tensor(
            pad_items(token_ids, train_data.vocab.padding_idx)[0], dtype=torch.long
        )
        subword_end_mask = torch.tensor(
            pad_items(subword_end_mask, 0)[0], dtype=torch.bool
        )

        return token_ids, action_ids, subword_end_mask, max_stack_size

    train_data = Dataset.from_json(
        args.train_file,
        args.batch_size,
        vocab=vocab,
        random_unk=args.random_unk,
        oracle=args.strategy,
        batch_group=args.batch_group,
        batch_token_size=args.batch_token_size,
        batch_action_size=args.batch_action_size,
        max_length_diff=args.max_group_length_diff,
        group_sentence_size=args.group_sentence_size,
    )
    train_dataset = DatasetForAccelerator(train_data.sents)

    sampler = []
    for idxs in train_data.length_to_idxs.values():
        idxs = list(np.random.permutation(idxs))
        sampler += idxs

    batch_sampler = list(
        torch.utils.data.BatchSampler(
            sampler, args.batch_size, drop_last=False
        )
    )
    batch_sampler = np.random.permutation(batch_sampler)

    class BatchSampler(torch.utils.data.BatchSampler):
        def __init__(self, sampler):
            self.sampler = sampler
            self.batch_size = args.batch_size
            self.drop_last = False

        def __len__(self):
            return len(self.sampler)

        def __iter__(self):
            for batch in self.sampler:
                yield batch

    batch_sampler = BatchSampler(batch_sampler)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        # num_workers=os.cpu_count(),
        # pin_memory=True,
    )

    vocab = train_data.vocab
    action_dict = train_data.action_dict
    val_data = Dataset.from_json(
        args.val_file,
        args.batch_size,
        vocab=vocab,
        action_dict=action_dict,
        oracle=args.strategy,
        batch_group=args.batch_group,
        batch_token_size=args.batch_token_size,
        batch_action_size=args.batch_action_size,
        max_length_diff=args.max_group_length_diff,
        group_sentence_size=args.group_sentence_size,
    )
    val_dataset = DatasetForAccelerator(val_data.sents)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        # num_workers=os.cpu_count(),
        # pin_memory=True,
    )
    vocab_size = int(train_data.vocab_size)
    if accelerator.is_local_main_process:
        logger.info(
            'Train: %d sents / %d batches, Val: %d sents / %d batches'
            % (
                len(train_data.sents),
                len(train_dataloader),
                len(val_data.sents),
                len(val_dataloader),
            )
        )
        logger.info('Vocab size: %d' % vocab_size)

    # if args.device == 'cuda' or args.device == 'parallel':
    #     device = 'cuda:{}'.format(args.gpu)
    # else:
    #     device = 'cpu'

    epoch = 1
    val_losses = []

    def is_early_stop():
        return (
            args.early_stop
            and len(val_losses) > max(10, args.early_stop_patience)
            and all(
                [
                    val_losses[-i - 2] < val_losses[-i - 1]
                    for i in range(args.early_stop_patience - 1)
                ]
            )
        )

    if args.train_from == '':
        model = create_model(args, action_dict, vocab)
        # if args.device == 'parallel' and torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        # model.to(device)
        optimizer = create_optimizer(args, model)
        scheduler = create_scheduler(args, optimizer)
    else:
        if accelerator.is_local_main_process:
            logger.info('Loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from, map_location=device)
        model = create_model(args, action_dict, vocab)
        # if args.device == 'parallel' and torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        # model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if accelerator.is_local_main_process:
            logger.info('Loading ')
        optimizer = create_optimizer(args, model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler = create_scheduler(args, optimizer)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler = create_scheduler(args, optimizer)
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
    if accelerator.is_local_main_process:
        logger.info("model architecture")
        logger.info(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model total parameters: {}'.format(total_params))
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    model.train()

    # if args.amp:
    #     scaler = torch.cuda.amp.GradScaler()

    global_batch_i = 0

    if accelerator.is_local_main_process:
        tb = TensorBoardLogger(args)
    while epoch <= args.num_epochs and not is_early_stop():
        start_time = time.time()
        if accelerator.is_local_main_process:
            logger.info('Starting epoch {}'.format(epoch))
        num_sents = 0.0
        num_words = 0.0
        num_actions = 0
        batch_i = 0
        total_a_ll = 0.0
        total_w_ll = 0.0
        prev_ll = 0.0
        batch_sizes = []

        def output_learn_log():
            param_norm = sum([p.norm() ** 2 for p in model.parameters()]).item() ** 0.5
            total_ll = total_a_ll + total_w_ll
            ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
            ll_diff = total_ll - prev_ll
            action_ppl = np.exp(-total_a_ll / num_actions)
            word_ppl = np.exp(-total_w_ll / num_words)

            # all_action_ppl = accelerator.gather(
            #     torch.tensor(action_ppl).to(device)
            # ).cpu().detach().numpy().mean()
            # all_word_ppl = accelerator.gather(torch.tensor(word_ppl).to(device)).cpu().detach().numpy().mean()
            # all_ppl = accelerator.gather(torch.tensor(ppl).to(device)).cpu().detach().numpy().mean()
            # all_ll_diff = accelerator.gather(torch.tensor(ll_diff).to(device)).cpu().detach().numpy().mean()
            # all_lr = accelerator.gather(
            #     torch.tensor(optimizer.param_groups[0]['lr']).to(device)
            # ).cpu().detach().numpy().mean()

            if accelerator.is_local_main_process:
                logger.info(
                    'Epoch: {}, Batch: {}/{}, LR: {:.4f}, '
                    'ActionPPL: {:.2f}, WordPPL: {:.2f}, '
                    'PPL: {:2f}, LL: {}, '
                    '|Param|: {:.2f}, E[batch size]: {}, Throughput: {:.2f} examples/sec'.format(
                        epoch,
                        batch_i,
                        len(train_dataloader),
                        optimizer.param_groups[0]['lr'],
                        action_ppl,
                        word_ppl,
                        ppl,
                        -ll_diff,
                        param_norm,
                        sum(batch_sizes) / len(batch_sizes),
                        num_sents / (time.time() - start_time),
                    )
                )
            return ppl, word_ppl, action_ppl

        def calc_loss(token_ids, action_ids, max_stack_size, subword_end_mask):
            loss, a_loss, w_loss = model(
                token_ids,
                action_ids,
                stack_size_bound=max_stack_size,
                subword_end_mask=subword_end_mask,
            )
            if args.loss_normalize == 'sum':
                loss = loss
            elif args.loss_normalize == 'batch':
                loss = loss / token_ids.size(0)
            elif args.loss_normalize == 'action':
                loss = loss / a_loss.size(0)
            elif args.loss_normalize == 'token':
                loss = loss / w_loss.size(0)
            return loss, a_loss, w_loss

        def batch_step(
            token_ids, action_ids, max_stack_size, subword_end_mask, num_divides
        ):
            optimizer.zero_grad()
            block_size = token_ids.size(0) // num_divides
            total_a_loss = 0
            total_w_loss = 0
            num_actions = 0
            num_words = 0
            for begin_idx in range(0, token_ids.size(0), block_size):
                end_idx = begin_idx + block_size
                div_token_ids = token_ids[begin_idx:end_idx]
                div_action_ids = action_ids[begin_idx:end_idx]
                div_subword_end_mask = subword_end_mask[begin_idx:end_idx]
                loss, a_loss, w_loss = calc_loss(
                    div_token_ids, div_action_ids, max_stack_size, div_subword_end_mask
                )
                if num_divides > 1:
                    loss = loss / num_divides

                accelerator.backward(loss)
                # if args.amp:
                #     scaler.scale(loss).backward()
                # else:
                #     loss.backward()

                total_a_loss += -a_loss.sum().detach().item()
                total_w_loss += -w_loss.sum().detach().item()
                num_actions += a_loss.size(0)
                num_words += w_loss.size(0)

                del loss
                del a_loss
                del w_loss
                gc.collect()
                torch.cuda.empty_cache()

            return total_a_loss, total_w_loss, num_actions, num_words

        def try_batch_step(
            token_ids, action_ids, max_stack_size, subword_end_mask, num_divides=1
        ):
            try:
                return batch_step(
                    token_ids, action_ids, max_stack_size, subword_end_mask, num_divides
                )
            except RuntimeError as e:  # memory error -> retry by reducing batch size
                # Error is processed outside this scope.
                # A hack to prevent memory leak when handling oov.
                # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
                msg = str(e)

            torch.cuda.empty_cache()
            logger.warning(msg)
            logger.warning(
                'Memory error occurs: token batch: {}, action batch: {}'.format(
                    token_ids.size(), action_ids.size()
                )
            )
            logger.warning('Retry by halfing batch sizes...')
            return try_batch_step(
                token_ids, action_ids, max_stack_size, subword_end_mask, num_divides * 2
            )

        for batch in train_dataloader:
            token_ids, action_ids, subword_end_mask, max_stack_size = batch
            batch_sizes.append(token_ids.size(0))
            # if args.device != 'parallel':
            #     token_ids = token_ids.to(device)
            #     action_ids = action_ids.to(device)
            #     subword_end_mask = subword_end_mask.to(device)

            batch_i += 1
            global_batch_i += 1
            # optimizer.zero_grad()

            batch_ll = try_batch_step(
                token_ids, action_ids, max_stack_size, subword_end_mask
            )
            total_a_ll += batch_ll[0]
            total_w_ll += batch_ll[1]

            # if args.amp:
            #     if args.max_grad_norm > 0:
            #        scaler.unscale_(optimizer)
            #         torch.nn.utils.clip_grad_norm_(
            #             model.parameters(), args.max_grad_norm
            #         )
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            #     if args.max_grad_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(
            #             model.parameters(), args.max_grad_norm
            #         )
            #     optimizer.step()

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm,
                )

            optimizer.step()
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            num_sents += token_ids.size(0)
            # assert token_ids.size(0) * token_ids.size(1) == w_loss.size(0)
            num_actions += batch_ll[2]
            num_words += batch_ll[3]

            # trying to obtain a discrete value that would have meaning at each epoch boundary.
            continuous_epoch = int(
                ((epoch - 1) + (batch_i / len(train_dataloader))) * 10000
            )
            if batch_i % args.print_every == 0:
                ppl, word_ppl, _ = output_learn_log()
                prev_ll = total_a_ll + total_w_ll
                if accelerator.is_local_main_process:
                    tb.write(
                        {'Train ppl': ppl, 'Train word ppl': word_ppl, 'lr': args.lr},
                        continuous_epoch,
                    )

            if (
                args.valid_every > 0
                and global_batch_i % args.valid_every == 0
                and accelerator.is_local_main_process
            ):
                do_valid(
                    model,
                    optimizer,
                    scheduler,
                    train_data,
                    val_dataloader,
                    tb,
                    epoch,
                    continuous_epoch,
                    val_losses,
                    accelerator,
                    args,
                )

        output_learn_log()
        epoch += 1
        if args.valid_every <= 0 and accelerator.is_local_main_process:
            do_valid(
                model,
                optimizer,
                scheduler,
                train_data,
                val_dataloader,
                tb,
                epoch,
                epoch,
                val_losses,
                accelerator,
                args,
            )

    # Last validation is necessary when validations were performed intermediately.
    if args.valid_every > 0 and accelerator.is_local_main_process:
        do_valid(
            model,
            optimizer,
            train_data,
            val_dataloader,
            tb,
            epoch,
            continuous_epoch,
            val_losses,
            accelerator,
            args,
        )
    if accelerator.is_local_main_process:
        logger.info("Finished training!")


def do_valid(
    model,
    optimizer,
    scheduler,
    train_data,
    val_dataloader,
    tb,
    epoch,
    step,
    val_losses,
    accelerator,
    args,
):
    best_val_loss = float('inf') if len(val_losses) == 0 else min(val_losses)
    # all_best_val_loss = accelerator.gather(
    #     torch.tensor(best_val_loss).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    logger.info('--------------------------------')
    logger.info('Checking validation perplexity...')
    val_loss, val_ppl, val_action_ppl, val_word_ppl = eval_action_ppl(
        val_dataloader, model, accelerator
    )

    # all_val_loss = accelerator.gather(
    #     torch.tensor(val_loss).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    # all_val_ppl = accelerator.gather(
    #     torch.tensor(val_ppl).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    # all_val_action_ppl = accelerator.gather(
    #     torch.tensor(val_action_ppl).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    # all_val_word_ppl = accelerator.gather(
    #     torch.tensor(val_word_ppl).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    # all_val_losses = accelerator.gather(
    #     torch.tensor(val_losses).to(accelerator.device)
    # ).cpu().detach().numpy().mean()
    logger.info(
        'PPL: {:2f}, Loss: {:2f}, ActionPPL: {:2f}, WordPPL: {:2f}'.format(
            val_ppl, val_loss, val_action_ppl, val_word_ppl
        )
    )
    tb.write(
        {
            'Valid ppl': val_ppl,
            'Valid action ppl': val_action_ppl,
            'Valid word ppl': val_word_ppl,
        },
        step,
    )
    tb.write({'Valid loss': val_loss}, use_time=True)
    logger.info('--------------------------------')
    # from apex import amp
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # accelerator.wait_for_everyone()
        checkpoint = {
            'args': args.__dict__,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': train_data.vocab,
            'prepro_args': train_data.prepro_args,
            'action_dict': train_data.action_dict,
            'epoch': epoch,
            'val_losses': val_losses,
        }
        logger.info('Saving checkpoint to {}'.format(args.save_path))
        torch.save(checkpoint, args.save_path)

    def consecutive_increase(target_losses):
        return len(target_losses) > 0 and all(
            target_losses[i] < target_losses[i + 1]
            for i in range(0, len(target_losses) - 1)
        )

    val_losses.append(val_loss)

    if epoch >= args.min_epochs and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)


def eval_action_ppl(data, model, accelerator):
    # device = next(model.parameters()).device
    model.eval()
    num_sents = 0
    num_words = 0
    num_actions = 0
    total_a_ll = 0
    total_w_ll = 0
    with torch.no_grad():
        for batch in data:
            token_ids, action_ids, subword_end_mask, max_stack_size = batch
            token_ids = token_ids.to(accelerator.device)
            action_ids = action_ids.to(accelerator.device)
            subword_end_mask = subword_end_mask.to(accelerator.device)
            max_stack_size = max_stack_size.to(accelerator.device)
            # if args.device != 'parallel':
            #     token_ids = token_ids.to(device)
            #     action_ids = action_ids.to(device)
            #     subword_end_mask = subword_end_mask.to(device)
            loss, a_loss, w_loss = model(
                token_ids,
                action_ids,
                stack_size_bound=max_stack_size,
                subword_end_mask=subword_end_mask,
            )
            total_a_ll += -a_loss.sum().detach().item()
            total_w_ll += -w_loss.sum().detach().item()

            num_sents += token_ids.size(0)
            num_words += w_loss.size(0)
            num_actions += a_loss.size(0)

    ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
    loss = -(total_a_ll + total_w_ll)
    action_ppl = np.exp(-total_a_ll / num_actions)
    word_ppl = np.exp(-total_w_ll / num_words)
    # if accelerator.is_local_main_process:
    #     logger.info(
    #         'PPL: {:2f}, Loss: {:2f}, ActionPPL: {:2f}, WordPPL: {:2f}'.format(
    #             ppl, loss, action_ppl, word_ppl
    #         )
    #     )

    model.train()
    return loss, ppl, action_ppl, word_ppl


if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler("{}.log".format(args.save_path)),
            logging.StreamHandler(),
        ],
    )

    main(args)
