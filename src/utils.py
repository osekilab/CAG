# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/utils.py

import re

import torch
from nltk import Tree


def get_in_order_actions(line, subword_tokenized=False, gpt2_tokenized=False):
    def _get_actions_recur(tree, actions=None):
        if actions is None:
            actions = []

        if len(tree) == 1:
            if isinstance(tree[0], str):  # preterminal
                actions.append('SHIFT')
            else:  # unary
                actions = _get_actions_recur(tree[0], actions)
                actions.append('NT({})'.format(tree.label()))
                actions.append('REDUCE')
        else:  # multiple children
            if subword_tokenized and isinstance(tree[0][0], str):
                # multiple pieces could be left_corner
                idx = 0
                while idx < len(tree) and '▁' not in tree[idx][0]:
                    idx += 1
                seg = idx + 1
            elif gpt2_tokenized and isinstance(tree[0][0], str):
                # multiple pieces could be left_corner
                idx = 0
                while (
                    idx + 1 < len(tree)
                    and isinstance(tree[idx + 1][0], str)
                    and 'Ġ' not in tree[idx + 1][0]
                ):
                    idx += 1
                seg = idx + 1
            else:
                seg = 1
            left_corners, others = tree[:seg], tree[seg:]

            for left_corner in left_corners:
                actions = _get_actions_recur(left_corner, actions)

            actions.append('NT({})'.format(tree.label()))

            for other in others:
                actions = _get_actions_recur(other, actions)
            actions.append('REDUCE')
        return actions

    tree = Tree.fromstring(line.strip())
    return _get_actions_recur(tree) + ['FINISH']


def get_top_down_max_stack_size(actions):
    stack = []
    max_size = 0
    for action in actions:

        if action == 'SHIFT':
            stack.append('w')

        elif action[:2] == 'NT':
            stack.append('(')

        elif action == 'REDUCE':
            while stack[-1] != '(':
                stack.pop()
            stack[-1] = 'w'

        max_size = max(max_size, len(stack))
    if len(stack) != 1:
        print(stack)
    assert len(stack) == 1
    return max_size


def get_in_order_max_stack_size(
    actions, tokens, subword_tokenized=False, gpt2_tokenized=False
):
    stack = []
    max_size = 0
    token_idx = 0
    for action in actions:

        if action == 'SHIFT':
            stack.append(tokens[token_idx])
            token_idx += 1

        elif action[:2] == 'NT':
            left_corner = [stack.pop()]
            if subword_tokenized:
                assert left_corner[0] == 1 or '▁' in left_corner[0]
                if left_corner[0] != 1:
                    # may need to further pop (left_corner may be multiple tokens)
                    while len(stack) > 0 and not (stack[-1] == 1 or '▁' in stack[-1]):
                        left_corner.append(stack.pop())
                    left_corner = left_corner[::-1]
            if gpt2_tokenized:
                if token_idx < len(tokens):
                    assert left_corner[0] == 1 or 'Ġ' in tokens[token_idx]
                if left_corner[0] != 1:
                    # may need to further pop (left_corner may be multiple tokens)
                    while len(stack) > 0 and not (
                        stack[-1] == 1 or 'Ġ' in left_corner[-1]
                    ):
                        left_corner.append(stack.pop())
                    left_corner = left_corner[::-1]
            stack.append('(')
            stack.extend(left_corner)

        elif action == 'REDUCE':
            while stack[-1] != '(':
                stack.pop()
            stack[
                -1
            ] = 1  # 0 means a constituent (not use a str to distinguish from tokens)
        max_size = max(max_size, len(stack))
    assert len(stack) == 1
    return max_size


def clean_number(word):
    new_word = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', word)
    return new_word


def pad_items(items, pad_id):
    """
    `items`: a list of lists (each row has different number of elements).
    Return:
        padded_items: a converted items where shorter rotoken are padded by pad_id.
        lengths: lengths of rotoken in original items.
    """
    lengths = [len(row) for row in items]
    max_length = max(lengths)
    for i in range(len(items)):
        items[i] = items[i] + ([pad_id] * (max_length - len(items[i])))
    return items, lengths


def berkeley_unk_conv(token):
    """This is a simplified version of unknown token conversion in BerkeleyParser.
    The full version is berkely_unk_conv2.
    """
    unk = "unk"
    last_char_idx = len(token) - 1
    second_last_char_idx = last_char_idx - 1
    third_last_char_idx = last_char_idx - 2
    after_last_char_idx = last_char_idx + 1
    if token[0].isupper():
        unk = "c" + unk
    if token[0].isdigit() and token[last_char_idx].isdigit():
        unk = unk + "n"
    elif last_char_idx <= 2:
        pass
    elif token[third_last_char_idx:after_last_char_idx] == "ing":
        unk = unk + "ing"
    elif token[second_last_char_idx:after_last_char_idx] == "ed":
        unk = unk + "ed"
    elif token[second_last_char_idx:after_last_char_idx] == "ly":
        unk = unk + "ly"
    elif token[last_char_idx] == "s":
        unk = unk + "s"
    elif token[third_last_char_idx:after_last_char_idx] == "est":
        unk = unk + "est"
    elif token[second_last_char_idx:after_last_char_idx] == "er":
        unk = unk + 'ER'
    elif token[third_last_char_idx:after_last_char_idx] == "ion":
        unk = unk + "ion"
    elif token[third_last_char_idx:after_last_char_idx] == "ory":
        unk = unk + "ory"
    elif token[0:2] == "un":
        unk = "un" + unk
    elif token[second_last_char_idx:after_last_char_idx] == "al":
        unk = unk + "al"
    else:
        for i in range(last_char_idx):
            if token[i] == '-':
                unk = unk + "-"
                break
            elif token[i] == '.':
                unk = unk + "."
                break
    return "<" + unk + ">"


def berkeley_unk_conv2(token):
    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False
    for char in token:
        if char.isdigit():
            hasDigit = True
        elif char == '-':
            hasDash = True
        elif char.isalpha():
            if char.islower():
                hasLower = True
            elif char.isupper():
                numCaps += 1
    result = 'UNK'
    lower = token.rstrip().lower()
    ch0 = token.rstrip()[0]
    if ch0.isupper():
        if numCaps == 1:
            result = result + '-INITC'
            # Remove this because it relies on a vocabulary, not given to this function (HN).
            # if lower in words_dict:
            #   result = result + '-KNOWNLC'
        else:
            result = result + '-CAPS'
    elif not (ch0.isalpha()) and numCaps > 0:
        result = result + '-CAPS'
    elif hasLower:
        result = result + '-LC'
    if hasDigit:
        result = result + '-NUM'
    if hasDash:
        result = result + '-DASH'
    if lower[-1] == 's' and len(lower) >= 3:
        ch2 = lower[-2]
        if not (ch2 == 's') and not (ch2 == 'i') and not (ch2 == 'u'):
            result = result + '-s'
    elif len(lower) >= 5 and not (hasDash) and not (hasDigit and numCaps > 0):
        if lower[-2:] == 'ed':
            result = result + '-ed'
        elif lower[-3:] == 'ing':
            result = result + '-ing'
        elif lower[-3:] == 'ion':
            result = result + '-ion'
        elif lower[-2:] == 'er':
            result = result + '-er'
        elif lower[-3:] == 'est':
            result = result + '-est'
        elif lower[-2:] == 'ly':
            result = result + '-ly'
        elif lower[-3:] == 'ity':
            result = result + '-ity'
        elif lower[-1] == 'y':
            result = result + '-y'
        elif lower[-2:] == 'al':
            result = result + '-al'
    return result


def get_subword_boundary_mask(tokens):
    if any('▁' in token for token in tokens):
        # subword-tokenized
        return ['▁' in token for token in tokens]
    elif any('Ġ' in token for token in tokens):
        subword_boundary_mask = ['Ġ' in tokens[i + 1] for i in range(len(tokens)-1)]
        return subword_boundary_mask + [True]
    else:
        return [True for token in tokens]


def bincount_and_supply(x, max_size):
    counts = x.bincount()
    assert counts.size(0) <= max_size
    if counts.size(0) < max_size:
        counts = torch.cat([counts, counts.new_zeros(max_size - counts.size(0))])
    return counts
