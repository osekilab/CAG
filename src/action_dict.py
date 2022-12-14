# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/action_dict.py

import torch


class TopDownActionDict:
    def __init__(self, nonterminals):
        assert isinstance(nonterminals, list)
        self.nonterminals = nonterminals
        self.id2action = ['<pad>', 'SHIFT', 'REDUCE'] + [
            'NT({})'.format(nonterminal) for nonterminal in nonterminals
        ]
        self.action2id = dict(
            [(action, _id) for _id, action in enumerate(self.id2action)]
        )
        self.padding_idx = 0

    def to_id(self, actions):
        return [self.action2id[action] for action in actions]

    def num_actions(self):
        return len(self.id2action)

    def num_nonterminals(self):
        return len(self.id2action) - 3

    def mask_shift(self, mask, batch_idx):
        mask[batch_idx][1] = 0

    def mask_reduce(self, mask, batch_idx):
        mask[batch_idx][2] = 0

    def mask_nonterminal(self, mask, batch_idx):
        mask[batch_idx][3:] = 0

    def is_pad(self, action):
        return action == 0

    def is_shift(self, action):
        return action == 1

    def is_reduce(self, action):
        return action == 2

    def is_nonterminal(self, action):
        return action > 2

    def nonterminal_id(self, action):
        return action - 3

    def nonterminal_begin_id(self):
        return 3

    def finish_action(self):
        return 2

    def make_action_tensor(self, action_strs, device='cpu'):
        action_ids = [
            [self.action2id[action] for action in action_str]
            for action_str in action_strs
        ]
        max_length = max([len(ids) for ids in action_ids])

        for i in range(len(action_ids)):
            action_ids[i] += [self.padding_idx] * (max_length - len(action_ids[i]))

        return torch.tensor(action_ids, device=device)

    def build_tree_str(self, actions, tokens, tags, subword_end_mask=None):
        tree_str = ''
        token_idx = 0
        subword_idx = 0
        for action in actions:
            if self.is_nonterminal(action):
                tree_str += ' ( {} '.format(
                    self.nonterminals[self.nonterminal_id(action)]
                )

            elif self.is_shift(action):
                if (subword_end_mask is None) or (
                    subword_end_mask is not None and subword_end_mask[subword_idx]
                ):
                    tree_str += ' ( {} {} ) '.format(tags[token_idx], tokens[token_idx])
                    token_idx += 1
                subword_idx += 1
            elif self.is_reduce(action):
                tree_str += ' ) '

        return tree_str.replace(' ( ', '(').replace(' ) ', ')').replace(')(', ') (')


class InOrderActionDict(TopDownActionDict):
    def __init__(self, nonterminals):
        super().__init__(nonterminals)
        self.id2action = ['<pad>', 'SHIFT', 'REDUCE', 'FINISH'] + [
            'NT({})'.format(nonterminal) for nonterminal in nonterminals
        ]
        self.action2id = dict(
            [(action, _id) for _id, action in enumerate(self.id2action)]
        )

    def mask_finish(self, mask, batch_idx):
        mask[batch_idx][3] = 0

    def mask_nonterminal(self, mask, batch_idx):
        mask[batch_idx][4:] = 0

    def is_finish(self, action):
        return action == 3

    def is_nonterminal(self, action):
        return action > 3

    def nonterminal_id(self, action):
        return action - 4

    def nonterminal_begin_id(self):
        return 4

    def finish_action(self):
        return 3

    def build_tree_str(self, actions, tokens, tags, subword_end_mask=None):
        stack = []
        token_idx = 0
        subword_idx = 0
        for action in actions:
            if self.is_nonterminal(action):
                top = stack.pop()
                stack.append(
                    ' ( {} '.format(self.nonterminals[self.nonterminal_id(action)])
                )
                stack.append(top)

            elif self.is_shift(action):
                if (subword_end_mask is None) or (
                    subword_end_mask is not None and subword_end_mask[subword_idx]
                ):
                    stack.append(
                        ' ( {} {} ) '.format(tags[token_idx], tokens[token_idx])
                    )
                    token_idx += 1
                subword_idx += 1
            elif self.is_reduce(action):
                open_idx = len(stack) - 1
                while not ('(' in stack[open_idx] and ')' not in stack[open_idx]):
                    # find until open elem (only '(' exists) is found
                    open_idx -= 1
                reduced = ''.join(stack[open_idx:] + [' ) '])
                stack = stack[:open_idx]
                stack.append(reduced)

        return (
            ''.join(stack).replace(' ( ', '(').replace(' ) ', ')').replace(')(', ') (')
        )
