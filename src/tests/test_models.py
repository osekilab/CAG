# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/test_models.py

import unittest

import torch
from numpy.testing import assert_almost_equal, assert_raises

from action_dict import TopDownActionDict
from models import FixedStackCompositionAttentionGrammar


class TestModels(unittest.TestCase):
    def test_composition_attention_grammar_transition(self):
        model = self._get_simple_top_down_model()
        self._test_transition_batch2(model)

    def _test_transition_batch2(self, model):
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        stack = model.build_stack(x, 100)

        trees = ["(S  (NP (NP 2 3 ) ) (NP 4 ) )", "(NP (NP 1   ) 2 5 )"]

        actions = self._trees_to_actions(trees)
        action_dict = model.action_dict
        actions = action_dict.make_action_tensor(actions)
        self.assertEqual(
            actions.cpu().numpy().tolist(),
            [[3, 4, 4, 1, 1, 2, 2, 4, 1, 2, 2], [4, 4, 1, 2, 1, 1, 2, 0, 0, 0, 0]],
        )

        word_vecs = model.emb(x)
        hidden_states = word_vecs.new_zeros(
            actions.size(1), word_vecs.size(0), model.hidden_size
        )

        initial_hidden = stack.hidden_head
        self.assertEqual(initial_hidden.size(), (2, model.hidden_size))
        hidden_states[0] = initial_hidden

        keys = []
        keys.append(stack.keys.clone())
        values = []
        values.append(stack.values.clone())
        trees = []
        trees.append(stack.trees.clone())

        hidden = model.transformer(word_vecs, actions[:, 0], stack)
        hidden_states[1] = hidden
        keys.append(stack.keys.clone())
        values.append(stack.values.clone())
        trees.append(stack.trees.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_index[:, :1], torch.tensor([[1], [1]])
        )

        nonterminals = [action_dict.nonterminal_id(i) for i in range(5)]
        self.assertTensorAlmostEqual(
            stack.nonterminal_ids[:, :1],
            torch.tensor([[nonterminals[3]], [nonterminals[4]]]),
        )
        self.assertTensorAlmostEqual(stack.nonterminal_index_pos, torch.tensor([0, 0]))

        hidden = model.transformer(word_vecs, actions[:, 1], stack)
        hidden_states[2] = hidden
        keys.append(stack.keys.clone())
        values.append(stack.values.clone())
        trees.append(stack.trees.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([2, 2]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_index[:, :2], torch.tensor([[1, 2], [1, 2]])
        )
        self.assertTensorAlmostEqual(
            stack.nonterminal_ids[:, :2],
            torch.tensor(
                [[nonterminals[3], nonterminals[4]], [nonterminals[4], nonterminals[4]]]
            ),
        )
        self.assertTensorAlmostEqual(stack.nonterminal_index_pos, torch.tensor([1, 1]))

        hidden = model.transformer(word_vecs, actions[:, 2], stack)
        hidden_states[3] = hidden
        keys.append(stack.keys.clone())
        values.append(stack.values.clone())
        trees.append(stack.trees.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_index[:, :3], torch.tensor([[1, 2, 3], [1, 2, 0]])
        )
        self.assertTensorAlmostEqual(
            stack.nonterminal_ids[:, :3],
            torch.tensor(
                [
                    [nonterminals[3], nonterminals[4], nonterminals[4]],
                    [nonterminals[4], nonterminals[4], 0],
                ]
            ),
        )
        self.assertTensorAlmostEqual(stack.nonterminal_index_pos, torch.tensor([2, 1]))

        hidden = model.transformer(word_vecs, actions[:, 3], stack)  # (SHIFT, REDUCE)
        hidden_states[4] = hidden
        keys.append(stack.keys.clone())
        values.append(stack.values.clone())
        trees.append(stack.trees.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([4, 2]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_index[:, :3], torch.tensor([[1, 2, 3], [1, 2, 0]])
        )
        self.assertTensorAlmostEqual(
            stack.nonterminal_ids[:, :3],
            torch.tensor(
                [
                    [nonterminals[3], nonterminals[4], nonterminals[4]],
                    [nonterminals[4], nonterminals[4], 0],
                ]
            ),
        )
        self.assertTensorAlmostEqual(stack.nonterminal_index_pos, torch.tensor([2, 0]))

        self.assertTrue(keys[3][1, 2].sum() != keys[4][1, 2].sum())
        self.assertTrue(values[3][1, 2].sum() != values[4][1, 2].sum())
        self.assertTrue(trees[3][1, 2].sum() != trees[4][1, 2].sum())

        hidden_states[5] = model.transformer(
            word_vecs, actions[:, 4], stack
        )  # (SHIFT, SHIFT)
        hidden_states[6] = model.transformer(
            word_vecs, actions[:, 5], stack
        )  # (REDUCE, SHIFT)
        hidden_states[7] = model.transformer(
            word_vecs, actions[:, 6], stack
        )  # (REDUCE, REDUCE)

        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(stack.nonterminal_index_pos, torch.tensor([0, -1]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_ids[:, :1],
            torch.tensor([[nonterminals[3]], [nonterminals[4]]]),
        )

        hidden_states[8] = model.transformer(
            word_vecs, actions[:, 7], stack
        )  # (NT(VP), <pad>)
        hidden_states[9] = model.transformer(
            word_vecs, actions[:, 8], stack
        )  # (SHIFT, <pad>)
        hidden_states[10] = model.transformer(
            word_vecs, actions[:, 9], stack
        )  # (REDUCE, <pad>)
        model.transformer(word_vecs, actions[:, 10], stack)  # (REDUCE, <pad>)

        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([1, 1]))

    def test_transformer_cell_with_beam_dim(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)
        stack = beam.stack
        self.assertEqual(stack.trees.size()[:2], (2, 2))
        batched_actions = [
            [[3, 4, 1, 1, 2], [4, 4, 1, 1, 2]],
            [[3, 1, 1, 1, 2], [3, 4, 4, 1, 2]],
        ]
        batched_actions = torch.tensor(batched_actions)
        for i in range(batched_actions.size(2)):
            a = batched_actions[..., i]
            model.transformer(word_vecs, a, stack)
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([[2, 2], [1, 3]]))
        self.assertTensorAlmostEqual(
            stack.nonterminal_index_pos, torch.tensor([[0, 0], [-1, 1]])
        )

    def test_beam_items_do_action(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)
        beam.beam_widths[:] = 2
        batched_actions = [
            [[3, 4, 1, 1, 2], [4, 4, 1, 1, 2]],
            [[3, 1, 1, 1, 2], [3, 4, 4, 4, 4]],
        ]
        batched_actions = torch.tensor(batched_actions)
        for i in range(batched_actions.size(2)):
            a = batched_actions[..., i]
            model.transformer(word_vecs, a, beam.stack)
            beam.do_action(a, model.action_dict)
        self.assertTensorAlmostEqual(beam.actions_pos, torch.tensor([[5, 5], [5, 5]]))
        self.assertTensorAlmostEqual(
            beam.actions[..., :6],
            torch.cat([torch.tensor([-1]).expand(2, 2, 1), batched_actions], 2),
        )
        self.assertTensorAlmostEqual(
            beam.num_constructed_nonterminals, torch.tensor([[0, 0], [0, 5]])
        )
        self.assertTensorAlmostEqual(
            beam.num_open_parentheses, torch.tensor([[1, 1], [0, 5]])
        )

    def test_beam_items_reconstruct(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 0, 0, 0, 0]))

        orig_hiddens = beam.stack.trees.clone()

        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        self.assertTensorAlmostEqual(new_beam_idx[0], torch.tensor([0, 0, 0, 1, 1]))
        self.assertTensorAlmostEqual(new_beam_idx[1], torch.tensor([0, 1, 2, 0, 1]))
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 2]))
        self.assertTensorAlmostEqual(beam.stack.trees[0, 0], orig_hiddens[0, 0])
        self.assertTensorAlmostEqual(beam.stack.trees[0, 0], beam.stack.trees[0, 1])
        self.assertTensorAlmostEqual(beam.stack.trees[0, 0], beam.stack.trees[0, 2])
        self.assertTensorNotEqual(beam.stack.trees[1, 0], beam.stack.trees[1, 2])

        orig_hiddens = beam.stack.trees.clone()
        actions = new_beam_idx[0].new_full((2, 3), 0)
        actions[new_beam_idx] = torch.tensor([3, 4, 4, 3, 4])  # [[S, NP, NP], [S, NP]]
        model.transformer(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 2]))
        self.assertTensorNotEqual(beam.stack.trees[0, 0], orig_hiddens[0, 1])
        self.assertTensorAlmostEqual(beam.stack.trees[0, 1], beam.stack.trees[0, 2])

        # second reconstruction
        reconstruct_idx = (
            torch.tensor([0, 0, 0, 1, 1, 1]),
            torch.tensor([2, 2, 0, 1, 1, 1]),
        )
        orig_hiddens = beam.stack.trees.clone()
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        self.assertTensorAlmostEqual(new_beam_idx[0], torch.tensor([0, 0, 0, 1, 1, 1]))
        self.assertTensorAlmostEqual(new_beam_idx[1], torch.tensor([0, 1, 2, 0, 1, 2]))
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(beam.stack.trees[0, 0], orig_hiddens[0, 2])
        self.assertTensorAlmostEqual(beam.stack.trees[0, 1], orig_hiddens[0, 2])
        self.assertTensorAlmostEqual(beam.stack.trees[0, 2], orig_hiddens[0, 0])
        self.assertTensorAlmostEqual(beam.stack.trees[1, 0], beam.stack.trees[1, 1])
        self.assertTensorAlmostEqual(beam.stack.trees[1, 1], beam.stack.trees[1, 1])
        self.assertTensorAlmostEqual(beam.stack.trees[1, 2], beam.stack.trees[1, 1])

    def test_beam_move_items(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (
            torch.tensor([0, 0, 0, 1, 1, 1]),
            torch.tensor([0, 0, 0, 0, 0, 0]),
        )
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        actions = torch.tensor([[3, 4, 1], [4, 1, 3]])  # (S, NP, shift); (NP, shift, S)
        model.transformer(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        empty_move_idx = (
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
        )
        word_completed_beam.move_items_from(beam, empty_move_idx, torch.tensor([]))
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([0, 0])
        )

        move_idx = (torch.tensor([0, 0, 1]), torch.tensor([1, 2, 1]))
        move_target_idx = word_completed_beam.move_items_from(
            beam, move_idx, torch.tensor([-0.5, -0.1, -0.2])
        )
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([2, 1])
        )
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([0, 0, 1]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([0, 1, 0]))
        self.assertBeamEqual(beam, word_completed_beam, move_idx, move_target_idx)
        self.assertTensorAlmostEqual(
            word_completed_beam.gen_ll[move_target_idx],
            torch.tensor([-0.5, -0.1, -0.2]),
        )
        self.assertTensorNotEqual(
            word_completed_beam.gen_ll[move_target_idx], beam.gen_ll[move_idx]
        )

        word_completed_beam.beam_size = 3  # reduce for test purpose.
        # Behavior when beam_widths exceed maximum.
        # Current beam width for word_completed_beam is [2, 1]; beam_size = 3
        # It will discard the last two elements if we try to add additional two elements to 0th batch.
        # Try to add (0, 0), (0, 2), (1, 0), (1, 2)
        move_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 0, 2]))
        move_target_idx = word_completed_beam.move_items_from(beam, move_idx)
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([3, 3])
        )
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([0, 1, 1]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([2, 1, 2]))
        reduced_move_idx = (
            torch.tensor([0, 1, 1]),
            torch.tensor([0, 0, 2]),
        )  # remove (0, 2) from move_idx
        self.assertBeamEqual(
            beam,
            word_completed_beam,
            reduced_move_idx,
            move_target_idx,
            check_scores=True,
        )

        # Further moving elements has no effects.
        move_idx = (torch.tensor([0, 1]), torch.tensor([2, 2]))
        move_target_idx = word_completed_beam.move_items_from(beam, move_idx)
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([3, 3])
        )
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([]))

    def test_beam_shrink(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0]))
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        actions = torch.tensor([[3, 4, 1], [4, 0, 0]])  # (S, NP, shift); (NP, shift, S)
        model.transformer(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        # First, make word_completed_beam a copy of beam.
        # After sorting, the top elements are [[1, 0], [0]]. (if shrinked size is 2)
        beam.gen_ll[reconstruct_idx] = torch.tensor([-4, -3, -10, -6]).float()
        word_completed_beam.move_items_from(beam, reconstruct_idx)

        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([3, 1])
        )
        word_completed_beam.shrink(2)
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([2, 1])
        )
        self.assertBeamEqual(
            beam,
            word_completed_beam,
            (torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])),
            (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])),
            check_scores=True,
        )

        # Further shrinking has no effects.
        word_completed_beam.shrink(2)
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([2, 1])
        )
        self.assertBeamEqual(
            beam,
            word_completed_beam,
            (torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])),
            (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])),
            check_scores=True,
        )

    def test_invalid_action_mask(self):
        model = self._get_simple_top_down_model(num_nonterminals=2)
        model.max_open_nonterminals = 5
        model.max_cons_nonterminals = 3
        x = torch.tensor([[2, 3], [1, 2]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)  # beam_size = 2
        sent_len = torch.tensor([2, 2])
        subword_end_mask = x != 0

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(  # (2, 2, 5); only nt is allowed.
                [
                    [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],  # beam idx 1 does not exist.
                    [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
                ]
            ),
        )

        reconstruct_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
        beam.reconstruct(reconstruct_idx)

        def do_action(actions):
            model.transformer(word_vecs, actions, beam.stack)
            beam.do_action(actions, model.action_dict)

        do_action(torch.tensor([[3, 3], [3, 3]]))  # (S, S); (S, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]], [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]]
            ),
        )

        do_action(torch.tensor([[3, 3], [3, 1]]))  # (S, S); (S, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]], [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]]
            ),
        )  # still reduce is prohibited (because this is not final token)

        do_action(torch.tensor([[3, 1], [3, 1]]))  # (S, shift); (S, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [
                    [[1, 0, 1, 1, 1], [1, 0, 0, 0, 0]],  # max_cons_nt = 3
                    [[1, 0, 1, 1, 1], [1, 1, 0, 1, 1]],
                ]
            ),
        )  # reduce is allowed; no shift word

        do_action(
            torch.tensor([[1, 2], [1, 2]])
        )  # (shift, r); (shift, r)  # (1, 1) finished
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 0, 0, 0], [1, 0, 1, 0, 0]], [[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]]
            ),
        )

        do_action(torch.tensor([[3, 1], [1, 0]]))  # (S, shift); (shift, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 1, 0, 0], [1, 1, 0, 1, 1]], [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]]
            ),
        )

        do_action(torch.tensor([[3, 2], [2, 0]]))  # (S, r); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [
                    [[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]],  # max_open_nonterminals = 5
                    [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]],
                ]
            ),
        )

        do_action(torch.tensor([[1, 0], [2, 0]]))  # (shift, -); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [
                    [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]],  # max_open_nonterminals = 5
                    [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]],
                ]
            ),
        )

        do_action(torch.tensor([[2, 0], [2, 0]]))  # (r, -); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [
                    [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]],  # max_open_nonterminals = 5
                    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],  # finished
                ]
            ),
        )

    def test_scores_to_successors(self):
        model = self._get_simple_top_down_model(num_nonterminals=2)
        model.max_open_nonterminals = 5
        model.max_cons_nonterminals = 3
        x = torch.tensor([[2, 3], [1, 2]])
        subword_end_mask = x != 0
        word_lengths = torch.tensor([2, 2])
        word_vecs = model.emb(x)
        beam_size, shift_size = 2, 1
        beam, word_completed_beam = model.build_beam_items(x, beam_size, shift_size)
        sent_len = torch.tensor([2, 2])

        def reconstruct_and_do_action(successors, word_completed_successors):
            if word_completed_successors[0][0].size(0) > 0:
                comp_idxs = tuple(word_completed_successors[0][:2])
                word_completed_beam.move_items_from(
                    beam, comp_idxs, new_gen_ll=word_completed_successors[2]
                )
            new_beam_idxs, _ = beam.reconstruct(successors[0][:2])
            beam.gen_ll[new_beam_idxs] = successors[2]
            actions = successors[1].new_full(
                (x.size(0), beam_size), model.action_dict.padding_idx
            )
            actions[new_beam_idxs] = successors[1]
            model.transformer(word_vecs, actions, beam.stack)
            beam.do_action(actions, model.action_dict)

        inf = -float('inf')
        scores = torch.tensor(
            [
                [[inf, inf, inf, -0.3, -0.5], [inf, inf, inf, inf, inf]],
                [[inf, inf, inf, -0.6, -0.2], [inf, inf, inf, inf, inf]],
            ]
        )

        succs, wc_succs, comps = model.scores_to_successors(
            x, word_lengths, 0, beam, scores, beam_size, shift_size
        )
        self.assertTensorAlmostEqual(succs[0][0], torch.tensor([0, 0, 1, 1]))
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 0, 0, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4, 4, 3]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.3, -0.5, -0.2, -0.6]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([]))
        self.assertTensorAlmostEqual(comps, torch.tensor([0, 0]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 2]))

        # This is actually the test for reconstruct_and_do_action defined above
        # (a part of model.beam_step).
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]], [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]]
            ),
        )

        scores = torch.tensor(
            [
                [
                    [
                        inf,
                        -0.4,
                        inf,
                        -0.6,
                        -0.8,
                    ],  # this shift is moved without fast-track.
                    [inf, -1.1, inf, -0.7, -0.9],
                ],  # no fast-track because shift_size = 1 and it is already consumed.
                [
                    [inf, -0.9, inf, -0.8, -0.6],  # this shift will be fast-tracked.
                    [inf, -1.4, inf, -0.7, -0.4],
                ],
            ]
        )  # this shift will not be saved.

        succs, wc_succs, comps = model.scores_to_successors(
            x, word_lengths, 0, beam, scores, beam_size, shift_size
        )
        self.assertTensorAlmostEqual(
            succs[0][0], torch.tensor([0, 1, 1])
        )  # shift (-0.1) is moved to wc_succs
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 1, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4, 4]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.6, -0.4, -0.6]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([-0.4, -0.9]))
        self.assertTensorAlmostEqual(comps, torch.tensor([0, 1]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([1, 2]))

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(
            mask,
            torch.tensor(
                [[[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]], [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]]
            ),
        )
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([1, 1])
        )

        scores = torch.tensor(
            [
                [
                    [
                        inf,
                        -1.2,
                        inf,
                        -0.8,
                        -1.0,
                    ],  # this shift is moved without fast-track.
                    [inf, inf, inf, inf, inf],
                ],  # no fast-track because shift_size = 1 and it is already consumed.
                [
                    [inf, inf, inf, inf, inf],  # to test finished batch.
                    [inf, inf, inf, inf, inf],
                ],
            ]
        )

        succs, wc_succs, comps = model.scores_to_successors(
            x, word_lengths, 0, beam, scores, beam_size, shift_size
        )
        self.assertTensorAlmostEqual(
            succs[0][0], torch.tensor([0, 0])
        )  # shift (-0.1) is moved to wc_succs
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.8, -1.0]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([0]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([0]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([1]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([-1.2]))
        self.assertTensorAlmostEqual(comps, torch.tensor([1, 0]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(
            word_completed_beam.beam_widths, torch.tensor([2, 1])
        )
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 0]))

        model.finalize_word_completed_beam(
            x,
            subword_end_mask,
            word_lengths,
            word_vecs,
            0,
            beam,
            word_completed_beam,
            2,
        )
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(
            beam.gen_ll[(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]))],
            torch.tensor([-0.4, -1.2, -0.9]),
        )

    def test_beam_search(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        subword_end_mask = x != 0
        parses, surprisals = model.word_sync_beam_search(x, subword_end_mask, 8, 5, 1)
        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)

        paths = set([tuple(parse) for parse, score in parses[0]])
        self.assertEqual(len(paths), 5)

        for parse, score in parses[0]:
            print([model.action_dict.id2action[action] for action in parse])
        print(surprisals[0])
        self.assertEqual([len(s) for s in surprisals], [3, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def test_beam_search_different_length(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4, 1, 3], [1, 2, 5, 0, 0]])
        subword_end_mask = x != 0
        parses, surprisals = model.word_sync_beam_search(x, subword_end_mask, 8, 5, 1)
        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)
        self.assertEqual(len(parses[1]), 5)

        for parse, score in parses[1]:
            print([model.action_dict.id2action[action] for action in parse])
        print(surprisals[1])
        self.assertEqual([len(s) for s in surprisals], [5, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[1]))

        for parse, score in parses[0]:
            self.assertEqual(len([a for a in parse if a == 1]), 5)  # 1 = shift
        for parse, score in parses[1]:
            self.assertEqual(len([a for a in parse if a == 1]), 3)

    def _trees_to_actions(self, trees):
        def conv(actions):
            if actions[0] == '(':
                return 'NT({})'.format(actions[1:])
            elif actions == ')':
                return 'REDUCE'
            else:
                return 'SHIFT'

        return [[conv(x) for x in tree.split()] for tree in trees]

    def _get_simple_top_down_model(
        self, vocab=6, w_dim=6, h_dim=6, num_layers=2, num_nonterminals=2, num_heads=2
    ):
        nonterminals = ['S', 'NP', 'VP', 'X3', 'X4', 'X5', 'X6'][:num_nonterminals]
        action_dict = TopDownActionDict(nonterminals)
        return FixedStackCompositionAttentionGrammar(
            action_dict,
            vocab_size=vocab,
            word_dim=w_dim,
            hidden_dim=h_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def assertTensorAlmostEqual(self, x, y):
        self.assertIsNone(
            assert_almost_equal(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        )

    def assertBeamEqual(self, beam1, beam2, idx1, idx2, check_scores=False):
        attrs = [
            'actions',
            'actions_pos',
            'num_constructed_nonterminals',
            'num_open_parentheses',
        ]
        if check_scores:
            attrs += ['gen_ll']
        stack_attrs = [
            'pointer',
            'top_position',
            'keys',
            'values',
            'trees',
            'hidden_head',
            'nonterminal_index',
            'nonterminal_ids',
            'nonterminal_index_pos',
        ]
        for attr in attrs:
            self.assertTensorAlmostEqual(
                getattr(beam1, attr)[idx1], getattr(beam2, attr)[idx2]
            )
        for attr in stack_attrs:
            self.assertTensorAlmostEqual(
                getattr(beam1.stack, attr)[idx1], getattr(beam2.stack, attr)[idx2]
            )

    def assertTensorNotEqual(self, x, y):
        self.assertIsNone(
            assert_raises(
                AssertionError,
                assert_almost_equal,
                x.cpu().detach().numpy(),
                y.cpu().detach().numpy(),
            )
        )
