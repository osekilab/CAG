# This script is based on the func†ion from https://github.com/IBM/transformers-struct-guidance/blob/main/src/plm-gen.py

import argparse

import numpy as np
import torch
from tqdm import tqdm

from utils import pad_items

from data import Dataset, DatasetForAccelerator

parser = argparse.ArgumentParser()

parser.add_argument('--test_file', default='data/ptb-no-unk-test.json')
parser.add_argument('--model_file', default='cag.pt')
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
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument(
    '--device',
    default='cuda',
    choices=['cuda', 'cpu'],
    help='If "cuda", GPU number --gpu is used.',
)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--strategy', default='top_down', choices=['top_down', 'in_order'])
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


def load_model(checkpoint, action_dict, vocab):
    if 'model_state_dict' in checkpoint:
        from train import create_model

        model = create_model(checkpoint['args'], action_dict, vocab)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        return checkpoint['model']


def main(args):
    if args.device == 'cuda':
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.model_file)
    vocab = checkpoint['vocab']
    action_dict = checkpoint['action_dict']
    repro_args = checkpoint['prepro_args']
    model = load_model(checkpoint, action_dict, vocab).to(device)

    if args.fp16:
        model.half()

    def collate_fn(batch):
        token_ids, action_ids, subword_end_mask, max_stack_size = list(zip(*batch))

        token_ids = list(token_ids)
        action_ids = list(action_ids)
        subword_end_mask = list(subword_end_mask)
        max_stack_size = max(list(max_stack_size))

        action_ids = torch.tensor(
            pad_items(action_ids, test_data.action_dict.padding_idx)[0],
            dtype=torch.long,
        )
        token_ids = torch.tensor(
            pad_items(token_ids, test_data.vocab.padding_idx)[0], dtype=torch.long
        )
        subword_end_mask = torch.tensor(
            pad_items(subword_end_mask, 0)[0], dtype=torch.bool
        )

        return token_ids, action_ids, subword_end_mask, max_stack_size

    test_data = Dataset.from_json(
        args.test_file,
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
    test_dataset = DatasetForAccelerator(test_data.sents)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        # num_workers=os.cpu_count(),
        # pin_memory=True,
    )

    def eval_action_ppl(data, model):
        device = next(model.parameters()).device
        model.eval()
        # num_sents = 0
        # num_words = 0
        # num_actions = 0
        total_a_ll = 0
        total_w_ll = 0
        word_count = 0
        with torch.no_grad():
            for batch in tqdm(data, total=len(data)):
                token_ids, action_ids, subword_end_mask, max_stack_size = batch
                token_ids = token_ids.to(device)
                action_ids = action_ids.to(device)
                subword_end_mask = subword_end_mask.to(device)
                max_stack_size = max_stack_size.to(device)
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

                word_count += subword_end_mask.sum().detach().item()

                # num_sents += token_ids.size(0)
                # num_words += w_loss.size(0)
                # num_actions += a_loss.size(0)

        # ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
        loss = -(total_a_ll + total_w_ll)
        # action_ppl = np.exp(-total_a_ll / num_actions)
        # word_ppl = np.exp(-total_w_ll / num_words)
        # if accelerator.is_local_main_process:
        #     logger.info(
        #         'PPL: {:2f}, Loss: {:2f}, ActionPPL: {:2f}, WordPPL: {:2f}'.format(
        #             ppl, loss, action_ppl, word_ppl
        #         )
        #     )

        # model.train()
        return np.exp(loss/word_count)

    ppl = eval_action_ppl(test_dataloader, model)
    print('Approximate word PPL: {}'.format(ppl))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
