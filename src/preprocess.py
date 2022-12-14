#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/preprocess.py

"""Create data files
"""

import argparse
import itertools
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import utils
from action_dict import InOrderActionDict, TopDownActionDict

from data import Vocabulary

pad = '<pad>'
unk = '<unk>'


def is_next_open_bracket(line, start_idx):
    start_idx += 1
    for char in line[start_idx:]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError(
        'Bracket possibly not balanced, open bracket not followed by closed bracket'
    )


def get_next_bracket_index(line, start_idx):
    for i in range(start_idx + 1, len(line)):
        char = line[i]
        if char == '(' or char == ')':
            return i
    raise IndexError(
        'Bracket possibly not balanced, open bracket not followed by closed bracket'
    )


def get_between_brackets(line, start_idx):
    start_idx += 1
    output = []
    for char in line[start_idx:]:
        if char == ')':
            break
        assert not (char == '(')
        output.append(char)
    return ''.join(output)


def get_tags_tokens_lowercase(line):
    output = []
    line = line.rstrip()
    for i in range(len(line)):
        if i == 0:
            assert line[i] == '('
        if line[i] == '(' and not (
            is_next_open_bracket(line, i)
        ):  # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line, i))

    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2  # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def transform_to_subword_tree(line, sp):
    line = line.rstrip()
    tags, tokens, _ = get_tags_tokens_lowercase(line)
    pieces = sp.encode(' '.join(tokens), out_type=str)
    end_idxs = [idx + 1 for idx, piece in enumerate(pieces) if '▁' in piece]
    begin_idxs = [0] + end_idxs[:-1]
    spans = list(
        zip(begin_idxs, end_idxs)
    )  # map from original token idx to piece span idxs.

    def _get_piece_preterms(tok_idx):
        tag = tags[tok_idx]
        begin_idx, end_idx = spans[tok_idx]
        span_pieces = pieces[begin_idx:end_idx]
        return ' '.join(['({} {})'.format(tag, piece) for piece in span_pieces])

    new_preterms = [_get_piece_preterms(idx) for idx in range(len(tokens))]
    orig_token_spans = []
    for i in range(len(line)):
        if line[i] == '(':
            next_bracket_idx = get_next_bracket_index(line, i)
            found_bracket = line[next_bracket_idx]
            if found_bracket == '(':
                continue  # not terminal -> skip
            orig_token_spans.append((i, next_bracket_idx + 1))
    assert len(new_preterms) == len(orig_token_spans)
    ex_span_ends = [span[0] for span in orig_token_spans] + [len(line)]
    ex_span_begins = [0] + [span[1] for span in orig_token_spans]
    parts = []
    for i in range(len(new_preterms)):
        start_idx = ex_span_begins[i]
        end_idx = ex_span_ends[i]
        parts.append(line[start_idx:end_idx])
        parts.append(new_preterms[i])
    start_idx = ex_span_begins[i + 1]
    end_idx = ex_span_ends[i + 1]
    parts.append(line[start_idx:end_idx])
    return ''.join(parts)


def transform_to_gpt2_tree(line, tokenizer):
    line = line.rstrip()
    tags, tokens, _ = get_tags_tokens_lowercase(line)

    pieces = []
    for token in tokens:
        if token not in list(tokenizer.added_tokens_decoder.values()):
            pieces += tokenizer.tokenize(token, add_prefix_space=True)
        else:
            pieces += ['Ġ' + token]

    end_idxs = [idx + 1 for idx in range(len(pieces) - 1) if 'Ġ' in pieces[idx + 1]]
    end_idxs += [len(pieces)]
    begin_idxs = [0] + end_idxs[:-1]
    spans = list(
        zip(begin_idxs, end_idxs)
    )  # map from original token idx to piece span idxs.

    def _get_piece_preterms(tok_idx):
        tag = tags[tok_idx]
        begin_idx, end_idx = spans[tok_idx]
        span_pieces = pieces[begin_idx:end_idx]
        return ' '.join(['({} {})'.format(tag, piece) for piece in span_pieces])

    new_preterms = [_get_piece_preterms(idx) for idx in range(len(tokens))]
    orig_token_spans = []
    for i in range(len(line)):
        if line[i] == '(':
            next_bracket_idx = get_next_bracket_index(line, i)
            found_bracket = line[next_bracket_idx]
            if found_bracket == '(':
                continue  # not terminal -> skip
            orig_token_spans.append((i, next_bracket_idx + 1))
    assert len(new_preterms) == len(orig_token_spans)
    ex_span_ends = [span[0] for span in orig_token_spans] + [len(line)]
    ex_span_begins = [0] + [span[1] for span in orig_token_spans]
    parts = []
    for i in range(len(new_preterms)):
        start_idx = ex_span_begins[i]
        end_idx = ex_span_ends[i]
        parts.append(line[start_idx:end_idx])
        parts.append(new_preterms[i])
    start_idx = ex_span_begins[i + 1]
    end_idx = ex_span_ends[i + 1]
    parts.append(line[start_idx:end_idx])
    return ''.join(parts)


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '('  # make sure it's an open bracket
    start_idx += 1
    output = []
    for char in line[start_idx:]:
        if char == ' ':
            break
        assert not (char == '(') and not (char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    idx = 0
    max_idx = len(line_strip) - 1
    while idx <= max_idx:
        assert line_strip[idx] == '(' or line_strip[idx] == ')'
        if line_strip[idx] == '(':
            if is_next_open_bracket(line_strip, idx):  # open non-terminal
                curr_NT = get_nonterminal(line_strip, idx)
                output_actions.append('NT(' + curr_NT + ')')
                idx += 1
                while (
                    line_strip[idx] != '('
                ):  # get the next open bracket, which may be a terminal or another non-terminal
                    idx += 1
            else:  # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[idx] != ')':
                    idx += 1
                idx += 1
                while line_strip[idx] != ')' and line_strip[idx] != '(':
                    idx += 1
        else:
            output_actions.append('REDUCE')
            if idx == max_idx:
                break
            idx += 1
            while line_strip[idx] != ')' and line_strip[idx] != '(':
                idx += 1
    assert idx == max_idx
    return output_actions


def find_nonterminals_in_tree(tree):
    tree = tree.strip()
    return re.findall(r'(?=\(([^\s]+)\s\()', tree)


def get_sent_info(arg):
    tree, setting = arg
    tree = tree.strip()
    lowercase, replace_num, vocab, sp, tokenizer, action_dict, io_action_dict = setting
    if sp is not None:
        # use sentencepiece
        tree = transform_to_subword_tree(tree, sp)
    elif tokenizer is not None:
        tree = transform_to_gpt2_tree(
            tree,
            tokenizer,
        )

    subword_tokenized = sp is not None
    gpt2_tokenized = tokenizer is not None
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    top_down_actions = get_actions(tree)
    in_order_actions = utils.get_in_order_actions(
        tree, subword_tokenized, gpt2_tokenized
    )
    top_down_max_stack_size = utils.get_top_down_max_stack_size(top_down_actions)
    assert len([action for action in in_order_actions if action == 'SHIFT']) == len(
        tokens
    )
    in_order_max_stack_size = utils.get_in_order_max_stack_size(
        in_order_actions,
        tokens,
        subword_tokenized,
        gpt2_tokenized,
    )
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    orig_tokens = tokens[:]
    if sp is None and tokenizer is None:
        # these are not applied with sentencepiece
        if lowercase:
            tokens = tokens_lower
        if replace_num:
            tokens = [utils.clean_number(token) for token in tokens]

        token_ids = [vocab.get_id(token) for token in tokens]
        converted_tokens = [vocab.id2word[token_id] for token_id in token_ids]

    elif sp is not None:
        token_ids = sp.piece_to_id(tokens)
        converted_tokens = tokens

    elif tokenizer is not None:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        converted_tokens = tokens

    top_down_action_ids = action_dict.to_id(top_down_actions)
    in_order_action_ids = io_action_dict.to_id(in_order_actions)

    return {
        'orig_tokens': orig_tokens,
        'tokens': converted_tokens,
        'token_ids': token_ids,
        'tags': tags,
        'actions': top_down_actions,
        'action_ids': top_down_action_ids,
        'max_stack_size': top_down_max_stack_size,
        'in_order_actions': in_order_actions,
        'in_order_action_ids': in_order_action_ids,
        'in_order_max_stack_size': in_order_max_stack_size,
        'tree_str': tree,
    }


def make_vocab(
    textfile,
    seqlength,
    minseqlength,
    lowercase,
    replace_num,
    vocabsize,
    vocabminfreq,
    unkmethod,
    jobs,
    apply_length_filter=True,
):
    word2count = defaultdict(int)
    with open(textfile, 'r') as f:
        trees = [tree.strip() for tree in f]
    with Pool(jobs) as pool:
        for tags, sent, sent_lower in pool.map(get_tags_tokens_lowercase, trees):
            assert len(tags) == len(sent)
            if lowercase:
                sent = sent_lower
            if replace_num:
                sent = [utils.clean_number(word) for word in sent]
            if (len(sent) > seqlength and apply_length_filter) or len(
                sent
            ) < minseqlength:
                continue

            for word in sent:
                word2count[word] += 1
    if unkmethod == 'berkeleyrule' or unkmethod == 'berkeleyrule2':
        conv_method = (
            utils.berkeley_unk_conv
            if unkmethod == 'berkeleyrule'
            else utils.berkeley_unk_conv2
        )
        berkeley_unks = set([conv_method(word) for word, _ in word2count.items()])
        specials = list(berkeley_unks)
    else:
        specials = [unk]
    if vocabminfreq:
        word2count = dict(
            [
                (word, count)
                for word, count in word2count.items()
                if count >= vocabminfreq
            ]
        )
    elif vocabsize > 0 and len(word2count) > vocabsize:
        sorted_wordcount = sorted(
            list(word2count.items()), key=lambda x: x[1], reverse=True
        )
        word2count = dict(sorted_wordcount[:vocabsize])
    return Vocabulary(list(word2count.items()), pad, unkmethod, unk, specials)


def learn_sentencepiece(textfile, output_prefix, args, apply_length_filter=True):
    import sentencepiece as spm

    with open(textfile, 'r') as f:
        trees = [tree.strip() for tree in f]
    user_defined_symbols = args.subword_user_defined_symbols or []
    if args.keep_ptb_bracket:
        user_defined_symbols += ['-LRB-', '-RRB-', '-LCB-', '-RCB-']
    with NamedTemporaryFile('wt') as tmp:
        with Pool(args.jobs) as pool:
            for tags, sent, sent_lower in pool.map(get_tags_tokens_lowercase, trees):
                assert len(tags) == len(sent)
                if (len(sent) > args.seqlength and apply_length_filter) or len(
                    sent
                ) < args.minseqlength:
                    continue
                tmp.write(' '.join(sent) + '\n')
        tmp.flush()
        spm.SentencePieceTrainer.train(
            input=tmp.name,
            model_prefix=output_prefix,
            vocab_size=args.vocabsize,
            model_type=args.subword_type,
            treat_whitespace_as_suffix=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=user_defined_symbols,
        )
    return spm.SentencePieceProcessor(model_file='{}.model'.format(output_prefix))


def get_gpt2tokenizer(args):
    from transformers import GPT2Tokenizer

    user_defined_symbols = args.subword_user_defined_symbols or []
    if args.keep_ptb_bracket:
        user_defined_symbols += ['-LRB-', '-RRB-', '-LCB-', '-RCB-']

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    num_added_tokens = tokenizer.add_tokens(
        ['Ġ' + symbol for symbol in user_defined_symbols]
    )

    return tokenizer, num_added_tokens


def get_data(args):
    def _get_nonterminals(textfiles, jobs=-1):
        nonterminals = set()
        for file_name in textfiles:
            with open(file_name, 'r') as f:
                lines = [line for line in f]
            with Pool(jobs) as pool:
                local_nonterminals = pool.map(find_nonterminals_in_tree, lines)
                nonterminals.update(
                    list(itertools.chain.from_iterable(local_nonterminals))
                )
        nonterminals = sorted(list(nonterminals))
        print('Found nonterminals: {}'.format(nonterminals))
        return nonterminals

    def _convert(
        textfile,
        lowercase,
        replace_num,
        seqlength,
        minseqlength,
        outfile,
        vocab,
        sp,
        tokenizer,
        action_dict,
        io_action_dict,
        apply_length_filter=True,
        jobs=-1,
    ):
        dropped = 0
        num_sents = 0
        conv_setting = (
            lowercase,
            replace_num,
            vocab,
            sp,
            tokenizer,
            action_dict,
            io_action_dict,
        )

        def _process_block(tree_with_settings, file_name):
            _dropped = 0
            with Pool(jobs) as pool:
                for sent_info in pool.map(get_sent_info, tree_with_settings):
                    tokens = sent_info['tokens']
                    if apply_length_filter and (
                        len(tokens) > seqlength or len(tokens) < minseqlength
                    ):
                        _dropped += 1
                        continue
                    sent_info['key'] = 'sentence'
                    file_name.write(json.dumps(sent_info) + '\n')
            return _dropped

        with open(outfile, 'wt') as f, open(textfile, 'r') as in_f:
            block_size = 100000
            tree_with_settings = []
            for tree in in_f:
                tree_with_settings.append((tree, conv_setting))
                if len(tree_with_settings) >= block_size:
                    dropped += _process_block(tree_with_settings, f)
                    num_sents += len(tree_with_settings)
                    tree_with_settings = []
                    print(num_sents)
            if len(tree_with_settings) > 0:
                _process_block(tree_with_settings, f)
                num_sents += len(tree_with_settings)

            others = {
                "vocab": vocab.to_json_dict() if vocab is not None else None,
                "nonterminals": nonterminals,
                "pad_token": pad,
                "unk_token": unk,
                "args": args.__dict__,
            }
            for key, value in others.items():
                f.write(json.dumps({'key': key, 'value': value}) + '\n')

        print(
            "Saved {} sentences (dropped {} due to length/unk filter)".format(
                num_sents, dropped
            )
        )

    print("First pass through data to get nonterminals...")
    nonterminals = _get_nonterminals(
        [args.trainfile, args.valfile, args.testfile], args.jobs
    )
    action_dict = TopDownActionDict(nonterminals)
    io_action_dict = InOrderActionDict(nonterminals)

    if args.unkmethod == 'subword':
        if args.vocabfile != '':
            print(
                'Loading pre-trained sentencepiece model from {}'.format(args.vocabfile)
            )
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=args.vocabfile)
            sp_model_path = '{}-spm.model'.format(args.outputfile)
            print('Copy sentencepiece model to {}'.format(sp_model_path))
            shutil.copyfile(args.vocabfile, sp_model_path)
        else:
            print(
                'unkmethod subword is selected. Running sentencepiece on the training data...'
            )
            sp = learn_sentencepiece(args.trainfile, args.outputfile + '-spm', args)
        vocab = None
        tokenizer = None
    elif args.unkmethod == 'gpt2':
        tokenizer, _ = get_gpt2tokenizer(args)
        tokenizer.save_pretrained(args.outputfile + '-tokenizer')
        vocab = None
        sp = None
    else:
        if args.vocabfile != '':
            print('Loading pre-specified source vocab from ' + args.vocabfile)
            vocab = Vocabulary.load(args.vocabfile)
        else:
            print("Second pass through data to get vocab...")
            vocab = make_vocab(
                args.trainfile,
                args.seqlength,
                args.minseqlength,
                args.lowercase,
                args.replace_num,
                args.vocabsize,
                args.vocabminfreq,
                args.unkmethod,
                args.jobs,
            )
        vocab.dump(args.outputfile + ".vocab")
        print("Vocab size: {}".format(len(vocab.id2word)))
        sp = None
        tokenizer = None

    _convert(
        args.testfile,
        args.lowercase,
        args.replace_num,
        0,
        args.minseqlength,
        args.outputfile + "-test.json",
        vocab,
        sp,
        tokenizer,
        action_dict,
        io_action_dict,
        0,
        args.jobs,
    )
    _convert(
        args.valfile,
        args.lowercase,
        args.replace_num,
        args.seqlength,
        args.minseqlength,
        args.outputfile + "-val.json",
        vocab,
        sp,
        tokenizer,
        action_dict,
        io_action_dict,
        0,
        args.jobs,
    )
    _convert(
        args.trainfile,
        args.lowercase,
        args.replace_num,
        args.seqlength,
        args.minseqlength,
        args.outputfile + "-train.json",
        vocab,
        sp,
        tokenizer,
        action_dict,
        io_action_dict,
        1,
        args.jobs,
    )


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--vocabsize',
        help="Size of vocabulary or subword vocabulary. "
        "When unkmethod is not subword, vocab is constructed "
        "by taking the top X most frequent words and "
        "rest are replaced with special UNK tokens. "
        "If unkmethod=subword, this defines the subword vocabulary size. ",
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--vocabminfreq',
        help="Minimum frequency for vocab. "
        "When this value > 0, this value is used instead of vocabsize.",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--unkmethod',
        help="How to replace an unknown token.",
        choices=['unk', 'berkeleyrule', 'berkeleyrule2', 'subword', 'gpt2'],
        default='unk',
    )
    parser.add_argument(
        '--subword_type',
        help="Segmentation algorithm in sentencepiece."
        "Note that --treat_whitespace_as_suffix for sentence_piece is always True.",
        choices=['bpe', 'unigram'],
        default='bpe',
    )
    parser.add_argument(
        '--keep_ptb_bracket',
        action='store_true',
        help='Recommended for English PTB-like dataset. Do not segment -LRB- and -RRB- into subwords.',
    )
    parser.add_argument(
        '--subword_user_defined_symbols',
        nargs='*',
        help='--user_defined_symbols for sentencepiece. These tokens are not segmented into subwords.',
    )
    parser.add_argument('--lowercase', help="Lower case", action='store_true')
    parser.add_argument(
        '--replace_num', help="Replace numbers with N", action='store_true'
    )
    parser.add_argument('--trainfile', help="Path to training data.", required=True)
    parser.add_argument('--valfile', help="Path to validation data.", required=True)
    parser.add_argument(
        '--testfile', help="Path to test validation data.", required=True
    )
    parser.add_argument(
        '--seqlength',
        help="Maximum sequence length. Sequences longer " "than this are dropped.",
        type=int,
        default=300,
    )
    parser.add_argument(
        '--minseqlength',
        help="Minimum sequence length. Sequences shorter " "than this are dropped.",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--outputfile',
        help="Prefix of the output file names. ",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--vocabfile',
        help="If working with a preset vocab, "
        "then including this will ignore srcvocabsize and use the"
        "vocab provided here. "
        "If unkmethod=subword, this argument specifies learned sentencepiece model.",
        type=str,
        default='',
    )
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args(arguments)
    if args.jobs == -1:
        args.jobs = os.cpu_count()
    # np.random.seed(3435)
    get_data(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
