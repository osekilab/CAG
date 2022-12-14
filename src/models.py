# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/fixed_stack_models.py

import math

import torch
import torch.nn.functional as F
import utils
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import GPT2Config, GPT2Model


class LSTMComposition(nn.Module):
    def __init__(self, dim, dropout):
        super(LSTMComposition, self).__init__()
        self.dim = dim
        self.rnn = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
        self.output = nn.Sequential(
            nn.Dropout(dropout, inplace=True), nn.Linear(dim * 2, dim), nn.ReLU()
        )

        self.batch_index = torch.arange(
            0, 10000, dtype=torch.long
        )  # cache with sufficient number.

    def forward(
        self, children, children_lengths, nonterminal, nonterminal_id, stack_state
    ):
        """
        :param children: (batch_size, max_num_children, input_dim)
        :param children_lengths: (batch_size)
        :param nonterminal: (batch_size, input_dim)
        :param nonterminal_id: (batch_size)
        """
        lengths = children_lengths + 2
        nonterminal = nonterminal.unsqueeze(1)

        elems = torch.cat([nonterminal, children, torch.zeros_like(nonterminal)], dim=1)
        batch_size = elems.size(0)
        elems[self.batch_index[:batch_size], lengths - 1] = nonterminal.squeeze(1)

        packed = pack_padded_sequence(
            elems, lengths.int().cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)

        gather_idx = (lengths - 2).unsqueeze(1).expand(-1, h.size(-1)).unsqueeze(1)

        unidirectional_size = self.dim
        fwd = h.gather(1, gather_idx).squeeze(1)[:, :unidirectional_size]
        bwd = h[:, 1, unidirectional_size:]
        c = torch.cat([fwd, bwd], dim=1)

        return self.output(c), None, None


class ExpandableStorage:
    def __init__(self):
        self.attrs = []

    def expand_at_dim(self, target_dim, new_size):
        def _same_dim_except_target(orig_size):
            if isinstance(orig_size, tuple):
                orig_size = list(orig_size)
            size_diff = new_size - orig_size[target_dim]
            orig_size[target_dim] = size_diff
            return orig_size

        for attr in self.attrs:
            old_x = getattr(self, attr)
            setattr(
                self,
                attr,
                torch.cat(
                    (old_x, old_x.new_zeros(_same_dim_except_target(old_x.size()))),
                    target_dim,
                ),
            )

    def expand_beam_dim(self, new_size):
        self.expand_at_dim(1, new_size)


class FixedStack(ExpandableStorage):
    def __init__(
        self,
        initial_emb,
        initial_hidden,
        initial_key,
        initial_value,
        stack_size,
        input_size,
        hidden_size,
        num_layers,
        num_heads,
        beam_size=1,
    ):
        super(FixedStack, self).__init__()
        device = initial_emb.device
        head_size = hidden_size // num_heads
        if head_size * num_heads != hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads\
                    (got `hidden_size`: {hidden_size} and `num_heads`: {num_heads})."
            )

        if beam_size == 1:
            batch_size = (initial_emb.size(0),)
            self.batch_index = (
                torch.arange(0, batch_size[0], dtype=torch.long, device=device),
            )
        else:
            batch_size = (initial_emb.size(0), beam_size)
            self.batch_index = (
                (
                    torch.arange(0, batch_size[0], dtype=torch.long, device=device)
                    .unsqueeze(1)
                    .expand(-1, beam_size)
                    .reshape(-1)
                ),
                torch.cat(
                    [
                        torch.arange(0, beam_size, dtype=torch.long, device=device)
                        for _ in range(batch_size[0])
                    ]
                ),
            )

        self.batch_size = initial_emb.size(0)
        self.stack_size = stack_size

        self.pointer = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )  # word pointer, b in the Noji and Oseki (2021)
        self.top_position = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )  # stack top position, p_h in the Noji and Oseki (2021)
        self.trees = initial_emb.new_zeros(
            batch_size + (stack_size + 1, input_size), device=device
        )
        self.keys = initial_emb.new_zeros(
            batch_size + (stack_size + 1, num_layers, num_heads, head_size),
            device=device,
        )
        self.values = initial_emb.new_zeros(
            batch_size + (stack_size + 1, num_layers, num_heads, head_size),
            device=device,
        )
        self.hidden_head = initial_hidden.new_zeros(
            batch_size + (input_size,), device=device
        )

        if beam_size == 1:
            self.trees[:, 0] = initial_emb
            self.keys[:, 0] = initial_key
            self.values[:, 0] = initial_value
            self.hidden_head[:] = initial_hidden
        else:
            # Only fill zero-th beam position because we do not have other beam elems at beginning of search.
            self.trees[:, 0, 0] = initial_emb
            self.keys[:, 0, 0] = initial_key
            self.values[:, 0, 0] = initial_value
            self.hidden_head[:, 0] = initial_hidden

        self.nonterminal_index = torch.zeros(
            batch_size + (stack_size,), dtype=torch.long, device=device
        )  # q in the Noji and Oseki (2021)
        self.nonterminal_ids = torch.zeros(
            batch_size + (stack_size,), dtype=torch.long, device=device
        )  # for Attention Composition
        self.nonterminal_index_pos = (
            torch.tensor([-1], dtype=torch.long, device=device)
            .expand(batch_size)
            .clone()
        )  # default is -1 (0 means zero-dim exists), p_q in the Noji and Oseki (2021)

        self.attrs = [
            'pointer',
            'top_position',
            'trees',
            'keys',
            'values',
            'hidden_head',
            'nonterminal_index',
            'nonterminal_ids',
            'nonterminal_index_pos',
        ]

    @property
    def beam_size(self):
        if self.trees.dim() == 4:
            return self.trees.size(1)
        else:
            return 1

    def reset_batch_index(self):
        # a necessary operation when expanding beam dimension.
        assert self.beam_size > 1
        device = self.batch_index[0].device
        self.batch_index = (
            (
                torch.arange(0, self.batch_size, dtype=torch.long, device=device)
                .unsqueeze(1)
                .expand(-1, self.beam_size)
                .reshape(-1)
            ),
            torch.cat(
                [
                    torch.arange(0, self.beam_size, dtype=torch.long, device=device)
                    for _ in range(self.batch_size)
                ]
            ),
        )

    def do_shift(self, shift_batches, shifted_embs, subword_end_mask=None):
        self.trees[
            shift_batches + (self.top_position[shift_batches] + 1,)
        ] = shifted_embs
        self.pointer[shift_batches] += 1
        self.top_position[shift_batches] += 1

    def do_nonterminal(self, nonterminal_batches, nonterminal_embs, nonterminal_ids):
        self.trees[
            nonterminal_batches + (self.top_position[nonterminal_batches] + 1,)
        ] = nonterminal_embs

        self.nonterminal_index_pos[nonterminal_batches] += 1
        self.nonterminal_ids[
            nonterminal_batches + (self.nonterminal_index_pos[nonterminal_batches],)
        ] = nonterminal_ids
        self.top_position[nonterminal_batches] += 1
        self.nonterminal_index[
            nonterminal_batches + (self.nonterminal_index_pos[nonterminal_batches],)
        ] = self.top_position[nonterminal_batches]

    def do_reduce(self, reduce_batches, new_child):
        prev_nonterminal_position = self.nonterminal_index[
            reduce_batches + (self.nonterminal_index_pos[reduce_batches],)
        ]
        self.trees[reduce_batches + (prev_nonterminal_position,)] = new_child
        self.nonterminal_index_pos[reduce_batches] -= 1
        self.top_position[reduce_batches] = prev_nonterminal_position

    def collect_reduced_children(self, reduce_batches):
        """
        :param reduce_batches: Tuple of idx tensors (output of nonzero())
        """
        nonterminal_index_pos = self.nonterminal_index_pos[reduce_batches]
        prev_nonterminal_position = self.nonterminal_index[
            reduce_batches + (nonterminal_index_pos,)
        ]
        reduced_nonterminal_ids = self.nonterminal_ids[
            reduce_batches + (nonterminal_index_pos,)
        ]
        reduced_nonterminals = self.trees[reduce_batches + (prev_nonterminal_position,)]
        child_length = self.top_position[reduce_batches] - prev_nonterminal_position
        max_child_length = child_length.max()

        child_idx = (
            prev_nonterminal_position.unsqueeze(1)
            + torch.arange(max_child_length, device=prev_nonterminal_position.device)
            + 1
        )
        child_idx[child_idx >= self.stack_size] = self.stack_size - 1
        # ceiled at maximum stack size (exceeding this may occur for some batches,
        # but those should be ignored safely.)
        child_idx = child_idx.unsqueeze(-1).expand(
            -1, -1, self.trees.size(-1)
        )  # (num_reduced_batch, max_num_child, input_dim)

        reduced_children = torch.gather(self.trees[reduce_batches], 1, child_idx)
        return (
            reduced_children,
            child_length,
            reduced_nonterminals,
            reduced_nonterminal_ids,
        )

    def update_hidden_key_value(self, hiddens, keys, values, target_position):
        position = self.top_position.reshape(-1).clone()
        self.hidden_head = hiddens
        self.keys[self.batch_index + (position,)] = keys[
            (torch.arange(target_position.size()[0]),) + (target_position,)
        ]
        self.values[self.batch_index + (position,)] = values[
            (torch.arange(target_position.size()[0]),) + (target_position,)
        ]

    def sort_by(self, sort_idx):
        """
        :param sort_idx: (batch_size, beam_size) or (batch_size)
        """

        def sort_tensor(tensor):
            _idx = sort_idx
            for i in range(sort_idx.dim(), tensor.dim()):
                _idx = _idx.unsqueeze(-1)
            return torch.gather(tensor, sort_idx.dim() - 1, _idx.expand(tensor.size()))

        for attr in self.attrs:
            old_x = getattr(self, attr)
            setattr(self, attr, sort_tensor(old_x))

    def move_beams(self, self_move_idxs, other, move_idxs):
        for attr in self.attrs:
            getattr(self, attr)[self_move_idxs] = getattr(other, attr)[move_idxs]


class CompositionAttentionGrammarCell(nn.Module):
    """
    CompositionAttentionGrammarCell receives next action and input word embedding, do action, and returns hidden states.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_heads,
        max_stack_size,
        dropout,
        embd_dropout,
        action_dict,
        attention_composition,
    ):
        super(CompositionAttentionGrammarCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.embd_dropout = nn.Dropout(embd_dropout, inplace=True)
        self.nonterminal_emb = nn.Sequential(
            nn.Embedding(action_dict.num_nonterminals(), input_size), self.embd_dropout
        )
        # self.nonterminal_emb = nn.Embedding(action_dict.num_nonterminals(), input_size)

        config = GPT2Config(
            n_positions=max_stack_size + 1,
            n_ctx=max_stack_size + 1,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.stack_transformer = GPT2Model(config)

        self.output = nn.Sequential(
            self.dropout, nn.Linear(hidden_size, input_size), nn.ReLU()
        )
        # self.output = nn.Linear(hidden_size, input_size, bias=False)
        self.composition = LSTMComposition(input_size, dropout)

        self.initial_emb = nn.Sequential(nn.Embedding(1, input_size), self.embd_dropout)
        # self.initial_emb = nn.Embedding(1, input_size)
        self.action_dict = action_dict

    def get_initial_emb_hidden_key_value(self, x):
        initial_emb = self.initial_emb(x.new_zeros(x.size(0), dtype=torch.long))
        (
            initial_hidden,
            initial_key,
            initial_value,
        ) = self.stack_transformer(inputs_embeds=initial_emb.unsqueeze(1))
        initial_hidden = initial_hidden.squeeze(1)
        initial_key = initial_key.squeeze(1)
        initial_value = initial_value.squeeze(1)
        return initial_emb, initial_hidden, initial_key, initial_value

    def forward(self, word_vecs, actions, stack, subword_end_mask=None):
        """
        Similar to update_stack_transformer.

        :param word_vecs: (batch_size, sent_len, input_size)
        :param actions: (batch_size, 1)
        """

        def _make_output(hidden_states, stack, position):
            new_output = hidden_states.new_zeros(
                stack.trees.size()[:-2] + (self.hidden_size,)
            )  # (batch_size, hidden_size)
            new_output[stack.batch_index] = hidden_states[
                (torch.arange(position.size()[0]),) + (position,)
            ]
            return new_output

        reduce_batches = (actions == self.action_dict.action2id['REDUCE']).nonzero(
            as_tuple=True
        )
        nonterminal_batches = (
            actions >= self.action_dict.nonterminal_begin_id()
        ).nonzero(as_tuple=True)
        shift_batches = (actions == self.action_dict.action2id['SHIFT']).nonzero(
            as_tuple=True
        )

        if shift_batches[0].size(0) > 0:
            shift_idx = (
                stack.pointer[shift_batches]
                .view(-1, 1, 1)
                .expand(-1, 1, word_vecs.size(-1))
            )
            shifted_embs = (
                torch.gather(word_vecs[shift_batches[0]], 1, shift_idx)
                .squeeze(1)
                .to(torch.float32)
            )
            stack.do_shift(shift_batches, shifted_embs, subword_end_mask)

        if nonterminal_batches[0].size(0) > 0:
            nonterminal_ids = (
                actions[nonterminal_batches] - self.action_dict.nonterminal_begin_id()
            )
            nonterminal_embs = self.nonterminal_emb(nonterminal_ids).to(torch.float32)
            stack.do_nonterminal(nonterminal_batches, nonterminal_embs, nonterminal_ids)

        if reduce_batches[0].size(0) > 0:
            stack_h = None

            (
                children,
                children_length,
                reduced_nonterminals,
                reduced_nonterminal_ids,
            ) = stack.collect_reduced_children(reduce_batches)

            new_child, _, _ = self.composition(
                children,
                children_length,
                reduced_nonterminals,
                reduced_nonterminal_ids,
                stack_h,
            )
            new_child = new_child.to(torch.float32)
            stack.do_reduce(reduce_batches, new_child)

        # Input for transformer should be (batch_size, sequence_length, input_size)
        # During beam search, stack.tree has different size.
        position = stack.top_position.reshape(-1).clone()
        max_top_position = position.max() + 1
        min_top_position = position.min()

        hidden_states, keys, values = self.stack_transformer(
            inputs_embeds=stack.trees.view(-1, stack.stack_size + 1, self.input_size)[
                :, min_top_position:max_top_position, :
            ],
            past_keys=stack.keys.view(
                -1,
                stack.stack_size + 1,
                self.num_layers,
                self.num_heads,
                self.head_size,
            )[:, :min_top_position, :, :, :],
            past_values=stack.values.view(
                -1,
                stack.stack_size + 1,
                self.num_layers,
                self.num_heads,
                self.head_size,
            )[:, :min_top_position, :, :, :],
        )
        target_position = position - min_top_position

        new_hidden = _make_output(hidden_states, stack, target_position)
        stack.update_hidden_key_value(new_hidden, keys, values, target_position)

        return stack.hidden_head


class FixedStackCompositionAttentionGrammar(nn.Module):
    def __init__(
        self,
        action_dict,
        vocab_size=100,
        padding_idx=0,
        word_dim=20,
        hidden_dim=20,
        num_layers=1,
        num_heads=1,
        max_stack_size=100,
        dropout=0,
        embd_dropout=0,
        attention_composition=False,
        max_open_nonterminals=100,
        max_cons_nonterminals=8,
    ):
        super(FixedStackCompositionAttentionGrammar, self).__init__()
        self.action_dict = action_dict
        self.padding_idx = padding_idx
        self.action_criterion = nn.CrossEntropyLoss(
            reduction='none', ignore_index=action_dict.padding_idx
        )

        self.word_criterion = nn.CrossEntropyLoss(
            reduction='none', ignore_index=padding_idx
        )

        self.dropout = nn.Dropout(embd_dropout, inplace=True)
        self.emb = nn.Sequential(
            nn.Embedding(vocab_size, word_dim, padding_idx=padding_idx), self.dropout
        )
        # self.emb = nn.Embedding(vocab_size, word_dim, padding_idx=padding_idx)

        self.transformer = CompositionAttentionGrammarCell(
            word_dim,
            hidden_dim,
            num_layers,
            num_heads,
            max_stack_size,
            dropout,
            embd_dropout,
            self.action_dict,
            attention_composition,
        )

        self.vocab_mlp = nn.Linear(word_dim, vocab_size)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_actions = action_dict.num_actions()  # num_labels + 2
        self.action_mlp = nn.Linear(word_dim, self.num_actions)
        self.input_size = word_dim
        self.hidden_size = hidden_dim
        self.vocab_mlp.weight = self.emb[0].weight
        # self.vocab_mlp.weight = self.emb.weight

        self.max_open_nonterminals = max_open_nonterminals
        self.max_cons_nonterminals = max_cons_nonterminals

    def forward(
        self, x, actions, initial_stack=None, stack_size_bound=-1, subword_end_mask=None
    ):
        assert isinstance(x, torch.Tensor)
        assert isinstance(actions, torch.Tensor)

        if stack_size_bound <= 0:
            stack_size = 100
        else:
            stack_size = stack_size_bound

        stack = self.build_stack(x, stack_size)
        word_vecs = self.emb(x)
        action_contexts = self.unroll_states(
            stack, word_vecs, actions, subword_end_mask
        )

        action_loss, _ = self.action_loss(actions, self.action_dict, action_contexts)
        word_loss, _ = self.word_loss(x, actions, self.action_dict, action_contexts)
        loss = action_loss.sum() + word_loss.sum()

        return loss, action_loss, word_loss

    def build_stack(self, x, stack_size=80):
        (
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
        ) = self.transformer.get_initial_emb_hidden_key_value(x)
        return FixedStack(
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
            stack_size,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
        )

    def unroll_states(self, stack, word_vecs, actions, subword_end_mask=None):

        hidden_states = word_vecs.new_zeros(
            actions.size(1), word_vecs.size(0), self.hidden_size
        )

        hidden_states[0] = stack.hidden_head

        for step in range(actions.size(1) - 1):
            hidden = self.transformer(
                word_vecs, actions[:, step], stack, subword_end_mask
            )  # (batch_size, input_size)
            hidden_states[step + 1] = hidden
        hidden_states = self.transformer.output(
            hidden_states.transpose(1, 0).contiguous()
        )  # (batch_size, action_len, input_size)
        # hidden_states = hidden_states.transpose(1, 0).contiguous()

        return hidden_states

    def action_loss(self, actions, action_dict, hiddens):
        assert hiddens.size()[:2] == actions.size()
        actions = actions.view(-1)
        hiddens = hiddens.view(actions.size(0), -1)

        action_mask = actions != action_dict.padding_idx
        idx = action_mask.nonzero(as_tuple=False).squeeze(1)
        actions = actions[idx]
        hiddens = hiddens[idx]

        logit = self.action_mlp(hiddens)
        loss = self.action_criterion(logit, actions)

        return loss, logit

    def word_loss(self, x, actions, action_dict, hiddens):
        assert hiddens.size()[:2] == actions.size()
        actions = actions.view(-1)
        hiddens = hiddens.view(actions.size(0), -1)

        action_mask = actions == action_dict.action2id['SHIFT']
        idx = action_mask.nonzero(as_tuple=False).squeeze(1)
        hiddens = hiddens[idx]

        x = x.view(-1)
        x = x[x != self.padding_idx]

        assert x.size(0) == hiddens.size(0)

        logit = self.vocab_mlp(hiddens)
        loss = self.word_criterion(logit, x)

        return loss, logit

    def word_sync_beam_search(
        self,
        x,
        subword_end_mask,
        beam_size,
        word_beam_size=0,
        shift_size=0,
        stack_size_bound=100,
        return_beam_history=False,
    ):
        self.eval()
        sent_lengths = (x != self.padding_idx).sum(dim=1)

        if (
            hasattr(self.transformer.composition, 'batch_index')
            and self.transformer.composition.batch_index.size(0) < x.size(0) * beam_size
        ):
            # The maximum number may be set by assuming training setting only.
            # Here we reset to the maximum number by beam search.
            self.transformer.composition.batch_index = torch.arange(
                0, x.size(0) * beam_size, device=x.device
            )

        if word_beam_size <= 0:
            word_beam_size = beam_size

        beam, word_completed_beam = self.build_beam_items(
            x, beam_size, shift_size, stack_size_bound=stack_size_bound
        )

        word_vecs = self.emb(x)
        word_marginal_ll = [[] for _ in range(x.size(0))]

        parses = [None] * x.size(0)
        surprisals = [[] for _ in range(x.size(0))]

        for pointer in range(x.size(1) + 1):
            forced_completions = x.new_zeros(x.size(0), dtype=torch.long)
            bucket_i = 0

            def _word_finished_batches():
                return (
                    (beam.beam_widths == 0)
                    + (  # Empty beam means no action remains (finished).
                        word_completed_beam.beam_widths >= word_completed_beam.beam_size
                    )
                    + (
                        (word_completed_beam.beam_widths - forced_completions)
                        >= beam_size
                    )
                )

            finished_batches = _word_finished_batches()

            while not finished_batches.all():
                added_forced_completions = self.beam_step(
                    x,
                    subword_end_mask,
                    sent_lengths,
                    word_vecs,
                    pointer,
                    beam,
                    word_completed_beam,
                    shift_size,
                )
                forced_completions += added_forced_completions
                finished_batches = _word_finished_batches()
                beam.beam_widths[
                    finished_batches.nonzero(as_tuple=True)
                ] = 0  # inactive word-finished batches.
                bucket_i += 1

            self.finalize_word_completed_beam(
                x,
                subword_end_mask,
                sent_lengths,
                word_vecs,
                pointer,
                beam,
                word_completed_beam,
                word_beam_size,
            )

            marginal = beam.marginal_probs()
            for batch, prob in enumerate(marginal.cpu().detach().numpy()):
                word_marginal_ll[batch].append(prob)

            for batch, length in enumerate(sent_lengths.cpu().detach()):
                if length == pointer:  # finished
                    for i in range(0, length):
                        surprisals[batch].append(
                            -word_marginal_ll[batch][i]
                            - (-word_marginal_ll[batch][i - 1] if i > 0 else 0)
                        )
                    parses[batch] = beam.nbest_parses(batch)
                    beam.beam_widths[batch] = 0  # inactivate finished batch

        ret = (parses, surprisals)
        return ret

    def finalize_word_completed_beam(
        self,
        x,
        subword_end_mask,
        sent_lengths,
        word_vecs,
        pointer,
        beam,
        word_completed_beam,
        word_beam_size,
    ):
        beam_size = word_completed_beam.beam_size
        word_completed_beam.shrink(word_beam_size)
        word_end_actions = x.new_full(
            (x.size(0), beam_size), self.action_dict.padding_idx
        )
        # active_idx = word_completed_beam.active_idxs()
        # if pointer < x.size(1):  # do shift
        #   word_end_actions[active_idx] = self.action_dict.action2id['SHIFT']
        # else:
        #   word_end_actions[active_idx] = self.action_dict.finish_action()
        active_idx_mask = word_completed_beam.active_idx_mask()
        shift_beam_idx_mask = (pointer < sent_lengths).unsqueeze(1) * active_idx_mask
        finish_beam_idx_mask = (pointer == sent_lengths).unsqueeze(1) * active_idx_mask
        word_end_actions[shift_beam_idx_mask] = self.action_dict.action2id['SHIFT']
        word_end_actions[finish_beam_idx_mask] = self.action_dict.finish_action()
        self.transformer(
            word_vecs, word_end_actions, word_completed_beam.stack, subword_end_mask
        )
        word_completed_beam.do_action(word_end_actions, self.action_dict)

        beam.clear()
        active_idx = active_idx_mask.nonzero(as_tuple=True)
        beam.move_items_from(word_completed_beam, active_idx)
        word_completed_beam.clear()

    def beam_step(
        self,
        x,
        subword_end_mask,
        sent_lengths,
        word_vecs,
        pointer,
        beam,
        word_completed_beam,
        shift_size,
    ):
        beam_size = beam.beam_size
        (
            successors,
            word_completed_successors,
            added_forced_completions,
        ) = self.get_successors(
            x,
            subword_end_mask,
            sent_lengths,
            pointer,
            beam,
            beam_size,
            shift_size,
        )

        # tuple of ((batch_idx, beam_idx), next_actions, total_scores)
        assert len(successors) == len(word_completed_successors) == 3
        if word_completed_successors[0][0].size(0) > 0:
            comp_idxs = tuple(word_completed_successors[0][:2])
            # Add elements to word_completed_beam
            # This assumes that returned scores are total scores rather than the current action scores.
            word_completed_beam.move_items_from(
                beam, comp_idxs, new_gen_ll=word_completed_successors[2]
            )

        new_beam_idxs, _ = beam.reconstruct(successors[0][:2])
        beam.gen_ll[new_beam_idxs] = successors[2]
        actions = successors[1].new_full(
            (x.size(0), beam_size), self.action_dict.padding_idx
        )
        actions[new_beam_idxs] = successors[1]
        self.transformer(word_vecs, actions, beam.stack, subword_end_mask)
        beam.do_action(actions, self.action_dict)

        return added_forced_completions

    def get_successors(
        self,
        x,
        subword_end_mask,
        sent_lengths,
        pointer,
        beam,
        beam_size,
        shift_size,
    ):
        if pointer < x.size(1):
            next_x = x[:, pointer]
        else:
            next_x = None

        invalid_action_mask = self.invalid_action_mask(
            beam, sent_lengths, subword_end_mask
        )  # (total beam size, n_actions)

        log_probs = self.action_log_probs(
            beam.stack, invalid_action_mask, next_x
        )  # (batch, beam, n_actions)

        # scores for inactive beam items (outside active_idx) are -inf on log_probs so we need
        # not worry about values in gen_ll outside active_idx
        log_probs += beam.gen_ll.unsqueeze(-1)

        return self.scores_to_successors(
            x,
            sent_lengths,
            pointer,
            beam,
            log_probs,
            beam_size,
            shift_size,
        )

    def scores_to_successors(
        self, x, sent_lengths, pointer, beam, total_scores, beam_size, shift_size
    ):
        num_actions = total_scores.size(2)
        total_scores = total_scores.view(total_scores.size(0), -1)
        sorted_scores, sort_idx = torch.sort(total_scores, descending=True)

        beam_id = sort_idx // num_actions
        action_id = sort_idx % num_actions

        valid_action_mask = sorted_scores != -float('inf')
        end_action_mask = (
            (pointer < sent_lengths).unsqueeze(1) * action_id
            == self.action_dict.action2id['SHIFT']
        ) + (
            (pointer == sent_lengths).unsqueeze(1)
            * self._parse_finish_mask(beam, action_id, beam_id)
        )

        end_action_mask = valid_action_mask * end_action_mask
        no_end_action_mask = valid_action_mask * (end_action_mask != 1)

        within_beam_end_action_idx = end_action_mask[:, :beam_size].nonzero(
            as_tuple=True
        )
        within_beam_num_end_actions = utils.bincount_and_supply(
            within_beam_end_action_idx[0], x.size(0)  # (batch_size)
        )

        # max num of forcefully shifted actions (actual number is upperbounded by active actions).
        num_to_be_forced_completions = torch.maximum(
            torch.tensor([0], device=x.device).expand(x.size(0)),
            shift_size - within_beam_num_end_actions,
        )  # (batch_size)

        outside_end_action_mask = end_action_mask[:, beam_size:]
        outside_end_action_idx = outside_end_action_mask.nonzero(as_tuple=True)
        # outside_end_action_idx[0] may be [0, 0, 0, 1, 2, 2]
        # num_forced_completions might be [0, 1, 0]
        # pick up only 4th element, i.e., make a mask [F, F, F, T, F, F]
        #
        # strategy:
        #   outside_num_end_actions: [3, 1, 2]
        #   order: [[0, 1, 2], [0, 1, 2], [0, 1, 2]
        #   size_recover_mask: [[T, T, T], [T, F, F], [T, T, F]]
        #   forced_completion_mask: [[F, F, F], [T, F, F], [F, F, F]]
        #   filter_mask: [F, F, F, T, F, F]
        outside_num_end_actions = utils.bincount_and_supply(
            outside_end_action_idx[0], x.size(0)
        )
        order = torch.arange(outside_num_end_actions.max(), device=x.device)
        size_recover_mask = order < outside_num_end_actions.unsqueeze(1)
        forced_completion_mask = order < num_to_be_forced_completions.unsqueeze(1)
        filter_mask = forced_completion_mask.view(-1)[size_recover_mask.view(-1)]
        outside_end_action_idx = (
            outside_end_action_idx[0][filter_mask],
            outside_end_action_idx[1][filter_mask],
        )

        outside_end_action_mask[:] = 0
        outside_end_action_mask[outside_end_action_idx] = 1

        num_forced_completions = utils.bincount_and_supply(
            outside_end_action_idx[0], x.size(0)
        )
        end_successor_idx = torch.cat(
            [end_action_mask[:, :beam_size], outside_end_action_mask], dim=1
        ).nonzero(as_tuple=True)
        no_end_successor_idx = no_end_action_mask[:, :beam_size].nonzero(as_tuple=True)

        def successor_idx_to_successors(successor_idx):
            next_beam_ids = beam_id[successor_idx]
            next_action_ids = action_id[successor_idx]
            next_scores = sorted_scores[successor_idx]
            return (successor_idx[0], next_beam_ids), next_action_ids, next_scores

        return (
            successor_idx_to_successors(no_end_successor_idx),
            successor_idx_to_successors(end_successor_idx),
            num_forced_completions,
        )

    def _parse_finish_mask(self, beam, action_id, beam_id):
        pre_final_mask = beam.num_open_parentheses.gather(1, beam_id) == 1
        end_action_mask = action_id == self.action_dict.finish_action()
        end_action_mask = end_action_mask * pre_final_mask
        return end_action_mask

    def action_log_probs(
        self,
        stack,
        invalid_action_mask,
        next_x=None,
        return_disc_probs=False,
    ):
        """
        :param stack: FixedStack
        :param invalid_action_mask: (batch_size, beam_size, num_actions) (inactive beams are entirely masked)
        :param next_x: (batch_size) to be shifted token ids
        """
        hiddens = self.transformer.output(
            stack.hidden_head.view(-1, self.hidden_size)
        )  # (beam*batch, hidden_size)
        # hiddens = stack.hidden_head.view(-1, self.hidden_size)
        # fp16 is cancelled here before softmax (I want to keep precision in final probabilities).
        action_logit = (
            self.action_mlp(hiddens).view(invalid_action_mask.size()).float()
        )  # (beam, batch, num_actions)
        action_logit[invalid_action_mask] = -float('inf')

        log_probs = F.log_softmax(
            action_logit, -1
        )  # (batch_size, beam_size, num_actions)
        log_probs[torch.isnan(log_probs)] = -float('inf')

        if return_disc_probs:
            disc_log_probs = log_probs.clone()

        if next_x is not None:  # shift is valid for next action
            word_logit = self.vocab_mlp(hiddens).float()  # (batch*beam, vocab_size)
            word_logit[:, self.padding_idx] = -float('inf')
            shift_idx = self.action_dict.action2id['SHIFT']
            next_x = (
                next_x.unsqueeze(1).expand(-1, log_probs.size(1)).clone().view(-1)
            )  # (batch*beam)
            word_log_probs = (
                self.word_criterion(word_logit, next_x) * -1.0
            )  # (batch_size*beam_size)
            word_log_probs = word_log_probs.view(log_probs.size(0), log_probs.size(1))
            log_probs[:, :, shift_idx] += word_log_probs

        if return_disc_probs:
            return (log_probs, disc_log_probs)
        else:
            return log_probs

    def invalid_action_mask(self, beam, sent_lengths, subword_end_mask):
        """
        Return a tensor where mask[i, j, k] = True means action k is not allowed for beam (i, j).
        """
        action_order = torch.arange(
            self.num_actions, device=beam.num_open_parentheses.device
        )

        sent_lengths = sent_lengths.unsqueeze(-1)  # add beam demension

        prev_pointer = beam.stack.pointer - 1
        prev_pointer[prev_pointer == -1] = 0
        prev_is_subword_mask = (beam.stack.pointer > 0) * (
            subword_end_mask.gather(1, prev_pointer) == 0
        )

        # reduce_mask[i, j, k] = True means k is not allowed reduce action for (i, j)
        reduce_mask = (action_order == self.action_dict.action2id['REDUCE']).view(
            1, 1, -1
        )
        reduce_mask = (
            ((beam.num_open_parentheses == 1) * (beam.stack.pointer < sent_lengths))
            +
            # prev is nt => cannot reduce immediately after nt
            (beam.prev_actions() >= self.action_dict.nonterminal_begin_id())
            + (beam.stack.top_position < 2)
            +
            # only shift is allowed when prev is subword
            prev_is_subword_mask
        ).unsqueeze(-1) * reduce_mask

        # nonterminal_mask[i, j, k] = True means k is not allowed nt action for (i, j).
        nonterminal_mask = (
            action_order >= self.action_dict.nonterminal_begin_id()
        ).view(1, 1, -1)
        nonterminal_mask = (
            (beam.num_open_parentheses >= self.max_open_nonterminals)
            + (beam.num_constructed_nonterminals >= self.max_cons_nonterminals)
            +
            # Check the storage of beam.actions, which is bounded beforehand.
            # Theoretically +1 seems sufficient (for rhs); extra +2 is for saving cases
            # where other actions (reduce/shift) are prohibited for some reasons.
            (
                beam.actions.size(2) - beam.actions_pos
                < (sent_lengths - beam.stack.pointer + beam.num_open_parentheses + 3)
            )
            +
            # Check the storage of fixed stack size (we need minimally two additional
            # elements to process arbitrary future structure).
            (beam.stack.top_position >= beam.stack.stack_size - 2)
            +
            # only shift is allowed when prev is subword
            prev_is_subword_mask
        ).unsqueeze(-1) * nonterminal_mask

        shift_mask = (action_order == self.action_dict.action2id['SHIFT']).view(
            1, 1, -1
        )
        shift_mask = (beam.stack.top_position >= beam.stack.stack_size - 1).unsqueeze(
            -1
        ) * shift_mask

        # all actions other than nonterminal are invalid;
        # except_nonterminal_mask[i, j, k] = True means k (not nonterminal) is not allowed for (i, j)
        except_nonterminal_mask = (
            action_order < self.action_dict.nonterminal_begin_id()
        ).view(1, 1, -1)
        except_nonterminal_mask = (beam.num_open_parentheses == 0).unsqueeze(
            -1
        ) * except_nonterminal_mask

        except_reduce_mask = (
            action_order != self.action_dict.action2id['REDUCE']
        ).view(1, 1, -1)
        except_reduce_mask = (beam.stack.pointer == sent_lengths).unsqueeze(
            -1
        ) * except_reduce_mask

        pad_mask = (action_order == self.action_dict.padding_idx).view(1, 1, -1)
        finished_mask = (
            (beam.stack.pointer == sent_lengths) * (beam.num_open_parentheses == 0)
        ).unsqueeze(-1)
        beam_width_mask = (
            torch.arange(beam.beam_size, device=reduce_mask.device).unsqueeze(0)
            >= beam.beam_widths.unsqueeze(1)
        ).unsqueeze(-1)

        return (
            reduce_mask
            + nonterminal_mask
            + shift_mask
            + except_nonterminal_mask
            + except_reduce_mask
            + pad_mask
            + finished_mask
            + beam_width_mask
        )

    def build_beam_items(
        self, x, beam_size, shift_size, particle=False, K=0, stack_size_bound=100
    ):

        if stack_size_bound <= 0:
            stack_size = 100
        else:
            stack_size = min(x.size(1) + 20, stack_size_bound)
        stack_size = math.ceil(stack_size / 8) * 8  # force to be multiple of 8.

        (
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
        ) = self.transformer.get_initial_emb_hidden_key_value(x)
        stack_unfinished, state_unfinished = self.new_beam_stack_with_state(
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
            stack_size,
            beam_size,
        )

        # The rationale behind (+shift_size*5) for beam size for finished BeamItem is
        # that # steps between words would probably be ~5 in most cases. Forcing to save shifts
        # after this many steps seems to be unnecessary.
        stack_word_finished, state_word_finished = self.new_beam_stack_with_state(
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
            stack_size,
            min(beam_size * 2, beam_size + shift_size * 5),
        )

        max_actions = max(100, x.size(1) * 5)

        return (
            BeamItems(
                stack_unfinished, state_unfinished, max_actions, False, particle, K
            ),
            BeamItems(
                stack_word_finished, state_word_finished, max_actions, True, particle, K
            ),
        )

    def new_beam_stack_with_state(
        self,
        initial_emb,
        initial_hidden,
        initial_key,
        initial_value,
        stack_size,
        beam_size,
    ):
        stack = FixedStack(
            initial_emb,
            initial_hidden,
            initial_key,
            initial_value,
            stack_size,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            beam_size,
        )

        stack_state = StackState(initial_emb.size(0), beam_size, initial_emb.device)

        return stack, stack_state


class StackState(ExpandableStorage):
    def __init__(self, batch_size, beam_size, device):
        """
        Keep track of information about states that is strategy-dependent, including
        num_constructed_nonterminals, for which how to update it will depend on the strategy.

        Structures other than FixedStack preserved in BeamItems would be
        strategy-invariant.
        """
        super(StackState, self).__init__()

        self.num_constructed_nonterminals = torch.zeros(
            (batch_size, beam_size), dtype=torch.long, device=device
        )
        self.num_open_parentheses = torch.zeros(
            (batch_size, beam_size), dtype=torch.long, device=device
        )

        self.attrs = ['num_constructed_nonterminals', 'num_open_parentheses']

    def move_beams(self, self_idxs, source, source_idxs):
        self.num_constructed_nonterminals[
            self_idxs
        ] = source.num_constructed_nonterminals[source_idxs]
        self.num_open_parentheses[self_idxs] = source.num_open_parentheses[source_idxs]

    def update_nonterminal_counts(self, actions, action_dict, action_path=None):
        shift_idxs = (actions == action_dict.action2id['SHIFT']).nonzero(as_tuple=True)
        nonterminal_idxs = (actions >= action_dict.nonterminal_begin_id()).nonzero(
            as_tuple=True
        )
        reduce_idxs = (actions == action_dict.action2id['REDUCE']).nonzero(
            as_tuple=True
        )

        self.num_constructed_nonterminals[shift_idxs] = 0
        self.num_open_parentheses[nonterminal_idxs] += 1
        self.num_constructed_nonterminals[nonterminal_idxs] += 1
        self.num_open_parentheses[reduce_idxs] -= 1
        self.num_constructed_nonterminals[reduce_idxs] = 0

    def sort_by(self, sort_idx):
        self.num_constructed_nonterminals = torch.gather(
            self.num_constructed_nonterminals, 1, sort_idx
        )
        self.num_open_parentheses = torch.gather(self.num_open_parentheses, 1, sort_idx)


class BeamItems(ExpandableStorage):
    def __init__(
        self,
        stack,
        stack_state,
        max_actions=500,
        beam_is_empty=False,
        particle_filter=False,
        initial_K=0,
    ):
        super(BeamItems, self).__init__()
        self.batch_size = stack.batch_size
        self.beam_size = stack.beam_size
        self.stack = stack
        self.stack_state = stack_state

        self.gen_ll = (
            torch.tensor([-float('inf')], device=stack.trees.device)
            .expand(
                self.batch_size,
                self.beam_size,
            )
            .clone()
        )
        self.gen_ll[:, 0] = 0

        if beam_is_empty:
            # how many beam elements are active for each batch?
            self.beam_widths = self.gen_ll.new_zeros(self.batch_size, dtype=torch.long)
        else:
            self.beam_widths = self.gen_ll.new_ones(self.batch_size, dtype=torch.long)

        self.action_path = ActionPath(
            self.batch_size, self.beam_size, max_actions, self.beam_widths.device
        )

        self.finished = self.beam_widths.new_zeros(
            (self.batch_size, self.beam_size),
        )

        self.attrs = ['gen_ll', 'finished']

    @property
    def num_constructed_nonterminals(self):
        return self.stack_state.num_constructed_nonterminals

    @property
    def num_open_parentheses(self):
        return self.stack_state.num_open_parentheses

    @property
    def actions(self):
        return self.action_path.actions

    @property
    def actions_pos(self):
        return self.action_path.actions_pos

    def prev_actions(self):
        return self.action_path.prev_actions()

    def nbest_parses(self, batch=None):
        return self.action_path.nbest_parses(self.beam_widths, self.gen_ll, batch)

    def shrink(self, size=-1):
        size = size if size > 0 else self.beam_size
        outside_beam_idx = (
            torch.arange(self.beam_size, device=self.gen_ll.device).unsqueeze(0)
            >= self.beam_widths.unsqueeze(1)
        ).nonzero(as_tuple=True)
        self.gen_ll[outside_beam_idx] = -float('inf')
        self.gen_ll, sort_idx = torch.sort(self.gen_ll, descending=True)
        self.stack.sort_by(sort_idx)
        self.stack_state.sort_by(sort_idx)
        self.action_path.sort_by(sort_idx)
        self.beam_widths = torch.min(
            self.beam_widths, self.beam_widths.new_tensor([size])
        )

    def active_idxs(self):
        """
        :return (batch_idxs, beam_idxs): All active idxs according to active beam sizes for each
                                        batch defined by self.beam_widths.
        """
        return self.active_idx_mask().nonzero(as_tuple=True)

    def active_idx_mask(self):
        order = torch.arange(self.beam_size, device=self.beam_widths.device)
        return order < self.beam_widths.unsqueeze(1)

    def clear(self):
        self.beam_widths[:] = 0

    def move_items_from(
        self, other, move_idxs, new_gen_ll=None, additional=(), allow_expand=False
    ):
        """
        :param other: BeamItems
        :param move_idxs: A pair of index tensors (for batch_index and beam_index)
        :param new_gen_ll: If not None, replace gen_ll of the target positions with this vector.
        :param additional: Tuple of vectors. If not empty, used to update AdditionalScores.
        """
        assert len(move_idxs) == 2  # hard-coded for beam search case.
        # This method internally presupposes that batch_index is sorted.
        assert torch.equal(move_idxs[0].sort()[0], move_idxs[0])
        move_batch_idxs, move_beam_idxs = move_idxs

        batch_numbers = utils.bincount_and_supply(move_batch_idxs, self.batch_size)
        max_moved_beam_size = batch_numbers.max()
        new_beam_widths = self.beam_widths + batch_numbers  # (batch_size)

        if new_beam_widths.max() >= self.beam_size:
            # The default case of handling beam widths exceeding max beam size, discarding
            # elements not fitted in self.beam_size.
            # This may be called even after the resize operation above because there is
            # an upperbound on the beam size.
            beam_idx_order = torch.arange(
                max_moved_beam_size, device=batch_numbers.device
            )
            sum_beam_idx_order = self.beam_widths.unsqueeze(1) + beam_idx_order
            move_idx_mask = sum_beam_idx_order < self.beam_size
            move_idx_mask = move_idx_mask.view(-1)[
                (beam_idx_order < batch_numbers.unsqueeze(1)).view(-1)
            ]
            move_idxs = (move_idxs[0][move_idx_mask], move_idxs[1][move_idx_mask])
            move_batch_idxs, move_beam_idxs = move_idxs
            if new_gen_ll is not None:
                new_gen_ll = new_gen_ll[move_idx_mask]
            batch_numbers = utils.bincount_and_supply(move_batch_idxs, self.batch_size)
            max_moved_beam_size = batch_numbers.max()
            new_beam_widths = self.beam_widths + batch_numbers  # (batch_size)

        self_move_beam_idxs = self.beam_widths.unsqueeze(1) + torch.arange(
            max_moved_beam_size, device=batch_numbers.device
        )
        self_beam_idx_mask = self_move_beam_idxs < new_beam_widths.unsqueeze(1)
        self_move_beam_idxs = self_move_beam_idxs.view(-1)[
            self_beam_idx_mask.view(-1).nonzero(as_tuple=True)
        ]
        assert self_move_beam_idxs.size() == move_beam_idxs.size()

        self_move_idxs = (move_batch_idxs, self_move_beam_idxs)
        self.beam_widths = new_beam_widths
        self._do_move_elements(other, self_move_idxs, move_idxs, new_gen_ll, additional)

        return self_move_idxs

    def reconstruct(self, target_idxs, allow_expand=False):
        """
        Intuitively perform beam[:] = beam[target_idxs]. target_idxs contains duplicates so this would
        copy some elements across different idxs. A core function in beam search.
        """
        assert self.beam_widths.sum() > 0

        assert len(target_idxs) == 2  # hard-coded for beam search case.
        move_batch_idxs, move_beam_idxs = target_idxs
        self.beam_widths = utils.bincount_and_supply(move_batch_idxs, self.batch_size)
        max_beam_widths = self.beam_widths.max()
        target_mask = None

        if max_beam_widths > self.beam_size:
            # need to shrink (may occur in particle filtering)
            beam_idx_order = torch.arange(max_beam_widths, device=target_idxs[0].device)
            target_mask = (
                beam_idx_order.unsqueeze(0).expand(self.batch_size, -1) < self.beam_size
            )
            target_mask = target_mask.view(-1)[
                (beam_idx_order < self.beam_widths.unsqueeze(1)).view(-1)
            ]
            target_idxs = (target_idxs[0][target_mask], target_idxs[1][target_mask])
            move_batch_idxs, move_beam_idxs = target_idxs
            self.beam_widths = utils.bincount_and_supply(
                move_batch_idxs, self.batch_size
            )
        assert self.beam_widths.max() <= self.beam_size

        self_move_beam_idxs = (
            torch.arange(self.beam_widths.max(), device=target_idxs[0].device)
            .unsqueeze(0)
            .repeat(self.beam_widths.size(0), 1)
        )
        self_beam_idx_mask = self_move_beam_idxs < self.beam_widths.unsqueeze(1)
        self_move_beam_idxs = self_move_beam_idxs.view(-1)[
            self_beam_idx_mask.view(-1).nonzero(as_tuple=True)
        ]

        assert self_move_beam_idxs.size() == move_beam_idxs.size()

        self_move_idxs = (move_batch_idxs, self_move_beam_idxs)
        self._do_move_elements(self, self_move_idxs, target_idxs)

        return self_move_idxs, target_mask

    def marginal_probs(self):
        active_idx_mask = self.active_idx_mask()
        self.gen_ll[active_idx_mask != 1] = -float('inf')
        return torch.logsumexp(self.gen_ll, 1)

    def _do_move_elements(
        self, source, self_idxs, source_idxs, new_gen_ll=None, new_additional=()
    ):
        self.gen_ll[self_idxs] = (
            new_gen_ll if new_gen_ll is not None else source.gen_ll[source_idxs]
        )
        self.stack_state.move_beams(self_idxs, source.stack_state, source_idxs)
        self.stack.move_beams(self_idxs, source.stack, source_idxs)
        self.action_path.move_beams(self_idxs, source.action_path, source_idxs)

    def do_action(self, actions, action_dict):
        # We need to use "unupdated" action_path for updating stack_state.
        self.stack_state.update_nonterminal_counts(
            actions, action_dict, self.action_path
        )
        self.action_path.add(actions, self.active_idxs())


class ActionPath(ExpandableStorage):
    def __init__(self, batch_size, beam_size, max_actions, device):
        super(ActionPath, self).__init__()
        self.actions = torch.full(
            (batch_size, beam_size, max_actions), -1, dtype=torch.long, device=device
        )
        self.actions_pos = self.actions.new_zeros(batch_size, beam_size)
        self.attrs = ['actions', 'actions_pos']

    def prev_actions(self):
        return self.actions.gather(2, self.actions_pos.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, beam_size)

    def move_beams(self, self_idxs, source, source_idxs):
        self.actions[self_idxs] = source.actions[source_idxs]
        self.actions_pos[self_idxs] = source.actions_pos[source_idxs]

    def add(self, actions, active_idxs):
        self.actions_pos[active_idxs] += 1
        self.actions[active_idxs + (self.actions_pos[active_idxs],)] = actions[
            active_idxs
        ]

    def sort_by(self, sort_idx):
        self.actions = torch.gather(
            self.actions, 1, sort_idx.unsqueeze(-1).expand(self.actions.size())
        )
        self.actions_pos = torch.gather(self.actions_pos, 1, sort_idx)

    def nbest_parses(self, beam_widths, gen_ll, tgt_batch=None):
        widths = beam_widths.cpu().numpy()
        actions = self.actions.cpu().numpy()
        actions_pos = self.actions_pos.cpu().numpy()

        def tree_actions(batch, beam):
            end = actions_pos[batch, beam] + 1
            return (
                actions[batch, beam, 1:end].tolist(),
                gen_ll[batch, beam].item(),
            )

        def batch_actions(batch):
            return [tree_actions(batch, i) for i in range(widths[batch])]

        if tgt_batch is not None:
            return batch_actions(tgt_batch)
        else:
            return [batch_actions(b) for b in range(len(widths))]
