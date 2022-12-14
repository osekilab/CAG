# This script is based on https://github.com/aistairc/rnng-pytorch/blob/master/data.py

import json
from collections import defaultdict

import numpy as np
import torch
from action_dict import InOrderActionDict, TopDownActionDict
from utils import (
    berkeley_unk_conv,
    berkeley_unk_conv2,
    clean_number,
    get_subword_boundary_mask,
    pad_items,
)


class Vocabulary(object):
    """
    This vocabulary prohibits registering a new token during lookup.
    Vocabulary should be constructed from a set of tokens with counts (word2count), a dictionary
    from a word to its count in the training data. (or anything)
    """

    def __init__(
        self,
        word2count_list,
        pad='<pad>',
        unkmethod='unk',
        unktoken='<unk>',
        specials=[],
    ):
        self.pad = pad
        self.padding_idx = 0
        self.specials = specials
        self.unkmethod = unkmethod
        self.unktoken = unktoken
        if self.unkmethod == 'unk':
            if unktoken not in specials:
                specials.append(unktoken)

        assert isinstance(word2count_list, list)
        self.id2word = [self.pad] + specials + [w for w, _ in word2count_list]
        self.word2id = dict([(word, id) for id, word in enumerate(self.id2word)])
        self.word2count = dict(word2count_list)
        self.id2count = dict(
            [(self.word2id[word], count) for word, count in self.word2count.items()]
        )

        if self.unkmethod == 'unk':
            self.unk_id = self.word2id[self.unktoken]

    def id_to_word(self, id):
        return self.id2word[id]

    def to_unk(self, word):
        if self.unkmethod == 'unk':
            return self.unktoken
        elif self.unkmethod == 'berkeleyrule':
            return berkeley_unk_conv(word)
        elif self.unkmethod == 'berkeleyrule2':
            return berkeley_unk_conv2(word)

    def to_unk_id(self, word_id):
        if self.unkmethod == 'unk':
            return self.unk_id
        else:
            if 1 <= word_id < 1 + len(self.specials):
                return word_id
            else:
                return self.get_id(berkeley_unk_conv(self.id2word[word_id]))

    def size(self):
        return len(self.id2word)

    def get_id(self, word):
        if word not in self.word2id:
            word = self.to_unk(word)
            if word not in self.word2id:
                # Back-off to a general unk token when converted unk-token is not registered in the
                # vocabulary (which happens when an unseen unk token is generated at test time).
                word = self.unktoken
        return self.word2id[word]

    def get_count_from_id(self, word_id):
        if word_id not in self.id2count:
            return 0
        else:
            return self.id2count[word_id]

    def get_count(self, word):
        if word not in self.word2count:
            return 0
        else:
            return self.word2count[word]

    # for serialization
    def list_word2count(self):
        first_word_idx = 1 + len(self.specials)
        return [(word, self.get_count(word)) for word in self.id2word[first_word_idx:]]

    def dump(self, file_name):
        with open(file_name, 'wt') as o:
            o.write(self.pad + '\n')
            o.write(self.unkmethod + '\n')
            o.write(self.unktoken + '\n')
            o.write(' '.join(self.specials) + '\n')
            for word, count in self.list_word2count():
                o.write('{}\t{}\n'.format(word, count))

    def to_json_dict(self):
        return {
            'pad': self.pad,
            'unkmethod': self.unkmethod,
            'unktoken': self.unktoken,
            'specials': self.specials,
            'word_count': self.list_word2count(),
        }

    @staticmethod
    def load(file_name):
        with open(file_name) as f:
            lines = [line for line in f]
        pad, unkmethod, unktoken, specials = [line.strip() for line in lines[:4]]
        specials = [word for word in specials]

        def _parse_line(line):
            word, count = line[:-1].split()
            return word, int(count)

        word2count_list = [_parse_line(line) for line in lines[4:]]
        return Vocabulary(word2count_list, pad, unkmethod, unktoken, specials)

    @staticmethod
    def from_data_json(data):
        vocab = data['vocab']
        return Vocabulary(
            vocab['word_count'],
            vocab['pad'],
            vocab['unkmethod'],
            vocab['unktoken'],
            vocab['specials'],
        )


class SentencePieceVocabulary(object):
    def __init__(self, sp_model_path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.padding_idx = self.sp.pad_id()
        self.pad = self.sp.id_to_piece(self.padding_idx)
        self.unkmethod = 'subword'
        self.unk_id = self.sp.unk_id()
        self.unktoken = self.sp.id_to_piece(self.unk_id)

    def id_to_word(self, id):
        return self.sp.id_to_piece(id)

    def to_unk(self, word):
        assert False, "SentencePieceVocabulary should not call to_unk()"

    def to_unk_id(self, word_id):
        assert False, "SentencePieceVocabulary should not call to_unk_id()"

    def size(self):
        return self.sp.get_piece_size()

    def get_id(self, word):
        return self.sp.piece_to_id(word)

    def get_count_from_id(self, word_id):
        return 1

    def get_count(self, word):
        return 1


class GPT2Vocabulary(object):
    def __init__(self, tokenizer_model_path):
        from transformers import GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_path)
        self.padding_idx = self.tokenizer.bos_token_id
        self.pad = self.tokenizer.bos_token
        self.unkmethod = 'gpt2'
        self.unktoken = self.tokenizer.unk_token

    def id_to_word(self, id):
        return self.tokenizer.convert_ids_to_tokens(id)

    def to_unk(self, word):
        assert False, "GPT2Vocabulary should not call to_unk()"

    def to_unk_id(self, word_id):
        assert False, "GPT2Vocabulary should not call to_unk_id()"

    def size(self):
        return len(self.tokenizer)

    def get_id(self, word):
        return self.tokenizer.convert_tokens_to_ids(word)

    def get_count_from_id(self, word_id):
        return 1

    def get_count(self, word):
        return 1


class Sentence(object):
    def __init__(
        self,
        orig_tokens,
        tokens,
        token_ids,
        tags,
        actions=None,
        action_ids=None,
        tree_str=None,
        max_stack_size=-1,
        is_subword_end=None,
    ):
        self.orig_tokens = orig_tokens
        self.tokens = tokens
        self.token_ids = token_ids
        self.tags = tags
        self.actions = actions or []
        self.action_ids = action_ids or []
        self.tree_str = tree_str  # original annotation
        self.max_stack_size = max_stack_size

        if is_subword_end is not None:
            assert isinstance(is_subword_end, list)
        else:
            assert self.tokens is not None
            is_subword_end = get_subword_boundary_mask(self.tokens)
        self.is_subword_end = is_subword_end

    @staticmethod
    def from_json(json_data, oracle='top_down'):
        if oracle == 'top_down':
            actions = json_data.get('actions', [])
            action_ids = json_data.get('action_ids', [])
        elif oracle == 'in_order':
            actions = json_data.get('in_order_actions', [])
            action_ids = json_data.get('in_order_action_ids', [])
        return Sentence(
            json_data['orig_tokens'],
            json_data['tokens'],
            json_data['token_ids'],
            json_data.get('tags', []),
            actions,
            action_ids,
            json_data.get('tree_str', None),
            json_data.get('max_stack_size', -1),
            json_data.get('is_subword_end', None),
        )

    def random_unked(self, vocab):
        def _unkify_rand(word_id):
            count = vocab.get_count_from_id(word_id)
            if count == 0 or (np.random.rand() < 1 / (1 + count)):
                return vocab.to_unk_id(word_id)
            else:
                return word_id

        return [_unkify_rand(token_id) for token_id in self.token_ids]

    # def get_word_end_mask(self):
    #   if any('▁' in t for t in self.tokens):
    #     # subword-tokenized
    #     return ['▁' in t for t in self.tokens]
    #   else:
    #     return [True for t in self.tokens]

    def get_subword_span_index(self):
        """
        Given tokens = ["In▁", "an▁", "Oct", ".▁", "19▁"],
        return [[0], [1], [2, 3], [4]].
        """
        idxs = []
        cur_word_idx = []
        for i, is_end in enumerate(self.is_subword_end):
            cur_word_idx.append(i)
            if is_end:
                idxs.append(cur_word_idx)
                cur_word_idx = []
        assert idxs[-1][-1] == len(self.tokens) - 1  # tokens is subword-segmented.
        return idxs

    def to_dict(self):
        return {
            'orig_tokens': self.orig_tokens,
            'tokens': self.tokens,
            'token_ids': self.token_ids,
            'tags': self.tags,
            'actions': self.actions,
            'action_ids': self.action_ids,
            'tree_str': self.tree_str,
            'max_stack_size': self.max_stack_size,
        }


class Dataset(object):
    def __init__(
        self,
        sents,
        batch_size,
        vocab,
        action_dict,
        random_unk=False,
        prepro_args={},
        batch_token_size=15000,
        batch_action_size=50000,
        batch_group='same_length',
        max_length_diff=20,
        group_sentence_size=1024,
    ):
        self.sents = sents
        self.batch_size = batch_size
        self.batch_token_size = (
            batch_token_size  # This bounds the batch size by the number of tokens.
        )
        self.batch_action_size = batch_action_size
        self.group_sentence_size = group_sentence_size
        self.vocab = vocab
        self.action_dict = action_dict
        self.random_unk = random_unk
        self.prepro_args = prepro_args  # keeps which process is performed.

        self.use_subwords = isinstance(self.vocab, SentencePieceVocabulary)
        self.use_tokenizer = isinstance(self.vocab, GPT2Vocabulary)

        self.vocab_size = vocab.size()
        if batch_group == 'same_length':
            self.length_to_idxs = self._get_length_to_idxs()
        elif batch_group == 'similar_length' or batch_group == 'similar_action_length':
            use_action_len = batch_group == 'similar_action_length'
            self.length_to_idxs = self._get_grouped_length_to_idxs(
                use_action_len=use_action_len, max_length_diff=max_length_diff
            )
        elif batch_group == 'random':
            self.length_to_idxs = self._get_random_length_to_idxs()
        self.num_batches = self.get_num_batches()

    @staticmethod
    def from_json(
        data_file,
        batch_size,
        vocab=None,
        action_dict=None,
        random_unk=False,
        oracle='top_down',
        batch_group='same_length',
        batch_token_size=15000,
        batch_action_size=50000,
        max_length_diff=20,
        group_sentence_size=1024,
    ):
        """If vocab and action_dict are provided, they are not loaded from data_file.
        This is for sharing these across train/valid/test sents.
        If random_unk = True, replace a token in a sentence to unk with a probability
        inverse proportional to the frequency in the training data.
        TODO: add custom unkifier?
        """

        def _new_action_dict(nonterminals):
            if oracle == 'top_down':
                return TopDownActionDict(nonterminals)
            elif oracle == 'in_order':
                return InOrderActionDict(nonterminals)

        json_data = Dataset._load_json_helper(data_file)
        sents = [
            Sentence.from_json(sentence, oracle) for sentence in json_data['sentences']
        ]
        vocab = vocab or Vocabulary.from_data_json(json_data)
        action_dict = action_dict or _new_action_dict(json_data['nonterminals'])

        return Dataset(
            sents,
            batch_size,
            vocab,
            action_dict,
            random_unk,
            json_data['args'],
            batch_group=batch_group,
            batch_token_size=batch_token_size,
            batch_action_size=batch_action_size,
            max_length_diff=max_length_diff,
            group_sentence_size=group_sentence_size,
        )

    @staticmethod
    def _load_json_helper(path):
        def _read_jsonl(file):
            data = {}
            sents = []
            for line in file:
                orig = json.loads(line)
                key = orig['key']
                if key == 'sentence':
                    orig['is_subword_end'] = get_subword_boundary_mask(orig['tokens'])
                    # Unused values are discarded here (for reducing memory for larger data).
                    orig['tokens'] = orig['orig_tokens'] = orig['tree_str'] = orig[
                        'actions'
                    ] = orig['in_order_actions'] = orig['tags'] = None
                    sents.append(orig)
                else:
                    # key except 'sentence' should only appear once
                    assert key not in data
                    data[key] = orig['value']
            data['sentences'] = sents
            return data

        try:
            with open(path) as f:
                # Old format => a single fat json object containing everything.
                return json.load(f)
        except json.decoder.JSONDecodeError:
            with open(path) as f:
                # New format => jsonl
                return _read_jsonl(f)

    @staticmethod
    def from_text_file(
        text_file,
        batch_size,
        vocab,
        action_dict,
        tagger_fn=None,
        prepro_args={},
        batch_token_size=15000,
        batch_group='same_length',
    ):
        """tagger_fn is a function receiving a sentence and returning POS tags.
        If Not provided, dummy tags (X) are provided.
        """
        tagger_fn = tagger_fn or (lambda tokens: ['X' for _ in tokens])
        sents = []
        with open(text_file) as f:
            for line in f:
                orig_tokens = line.strip().split()
                tokens = orig_tokens[:]
                if isinstance(vocab, SentencePieceVocabulary):
                    tokens = vocab.sp.encode(' '.join(tokens), out_type=str)
                    token_ids = vocab.sp.piece_to_id(tokens)
                elif isinstance(vocab, GPT2Vocabulary):
                    pieces = []
                    for token in tokens:
                        if token not in list(
                            vocab.tokenizer.added_tokens_decoder.values()
                        ):
                            pieces += vocab.tokenizer.tokenize(
                                token, add_prefix_space=True
                            )
                        else:
                            pieces += ['Ġ' + token]

                    tokens = pieces
                    token_ids = vocab.tokenizer.convert_tokens_to_ids(tokens)

                else:
                    if prepro_args.get('lowercase', False):
                        tokens = [token.lower() for token in tokens]
                    if prepro_args.get('replace_num', False):
                        tokens = [clean_number(token) for token in tokens]
                    token_ids = [vocab.get_id(token) for token in tokens]
                tags = tagger_fn(orig_tokens)
                sent = Sentence(orig_tokens, tokens, token_ids, tags)
                sents.append(sent)
        return Dataset(
            sents,
            batch_size,
            vocab,
            action_dict,
            False,
            prepro_args,
            batch_token_size,
            batch_group=batch_group,
        )

    def get_num_batches(self):
        num_batches = 0
        for _, idxs in self.length_to_idxs.items():
            if len(idxs) % self.batch_size == 0:
                num_batches += len(idxs) // self.batch_size
            else:
                num_batches += (len(idxs) // self.batch_size) + 1
        return num_batches

    def batches(self, shuffle=True):
        yield from self.batches_helper(self.length_to_idxs, shuffle)

    def test_batches(self, block_size=1000, max_length_diff=20):
        assert block_size > 0
        """
        Sents are first segmented (chunked) by block_size, and then, mini-batched.
        Since each batch contains batch_idx, we can recover the original order of
        data, by processing output grouping this size.
        This may be useful when outputing the parse results (approximately) streamly,
        by dumping to stdout (or file) at once for every 100~1000 sentences.
        Below is an such example to dump parse results by keeping the original sentence
        order.
        ```
        batch_size = 3
        block_size = 1000
        parses = []
        idxs = []
        for token, action, idx in dataset.test_batches(block_size):
            parses.extend(parse(token))
            idxs.extend(idx)
            if len(idxs) >= block_size:
                assert len(idxs) <= block_size
                parse_idx_to_sent_idx = sorted(list(enumerate(idxs)), key=lambda x:x[1])
                orig_order_parses = [
                    parses[sent_idx] for (parse_idx, sent_idx) in parse_idx_to_sent_idx
                ]
                # process orig_order_parses (e.g., print)
                parses = []
                idxs = []
            ```
        """
        for offset in range(0, len(self.sents), block_size):
            end = min(len(self.sents), offset + block_size)
            length_to_idxs = self._get_grouped_length_to_idxs(
                range(offset, end), max_length_diff=max_length_diff
            )
            yield from self.batches_helper(length_to_idxs, False, True)

    def batches_helper(self, length_to_idxs, shuffle=True, test=False):
        # `length_to_idxs` summarizes sentence length to idx in `self.sents`.
        # This may be a subset of sentences, or full sentences.
        batches = []
        for length, idxs in length_to_idxs.items():
            if shuffle:
                idxs = np.random.permutation(idxs)

            def _add_batch(begin, end):
                assert begin < end
                batches.append(idxs[begin:end])

            longest_sent_len = 0
            longest_action_len = 0
            in_batch_idx = 0  # for i-th batch
            start_idx = 0
            batch_token_size = self.batch_token_size
            batch_action_size = self.batch_action_size
            # Create each batch to guarantee that (batch_size*max_sent_len) does not exceed
            # batch_token_size.
            for i in range(len(idxs)):
                cur_sent_len = len(self.sents[idxs[i]].token_ids)
                cur_action_len = len(self.sents[idxs[i]].action_ids)
                longest_sent_len = max(longest_sent_len, cur_sent_len)
                longest_action_len = max(longest_action_len, cur_action_len)
                if len(self.sents[idxs[i]].token_ids) > 100:
                    # Long sequence often requires larger memory and tend to cause memory error.
                    # Here we try to reduce the elements in a batch for such sequences, considering
                    # that they are rare and will not affect the total speed much.
                    batch_token_size = int(self.batch_token_size * 0.7)
                    batch_action_size = int(self.batch_action_size * 0.7)
                if i > start_idx and (  # for ensuring batch size 1
                    (longest_sent_len * (in_batch_idx + 1) >= batch_token_size)
                    or (longest_action_len * (in_batch_idx + 1) >= batch_action_size)
                    or (in_batch_idx > 0 and in_batch_idx % self.batch_size == 0)
                ):
                    _add_batch(start_idx, i)
                    in_batch_idx = 0  # i is not included in prev batch
                    longest_sent_len = cur_sent_len
                    longest_action_len = cur_action_len
                    start_idx = i
                    batch_token_size = self.batch_token_size
                    batch_action_size = self.batch_action_size
                in_batch_idx += 1
            _add_batch(start_idx, i + 1)
        self.num_batches = len(batches)

        if shuffle:
            batches = np.random.permutation(batches)

        if self.random_unk:

            def _conv_sent(idx):
                return self.sents[idx].random_unked(self.vocab)

        else:

            def _conv_sent(idx):
                return self.sents[idx].token_ids

        for batch_idxs in batches:
            token_ids = [_conv_sent(idx) for idx in batch_idxs]
            tokens = torch.tensor(self._pad_token_ids(token_ids), dtype=torch.long)
            ret = (tokens,)
            if not test:
                action_ids = [self.sents[idx].action_ids for idx in batch_idxs]
                max_stack_size = max(
                    [self.sents[idx].max_stack_size for idx in batch_idxs]
                )
                ret += (
                    torch.tensor(self._pad_action_ids(action_ids), dtype=torch.long),
                    max_stack_size,
                )
            if self.use_subwords or self.use_tokenizer:
                subword_end_mask = self.get_subword_end_mask(batch_idxs)
            else:
                subword_end_mask = torch.full(tokens.size(), 1, dtype=torch.bool)
            ret += (
                subword_end_mask,
                batch_idxs,
            )
            yield ret

    def get_subword_end_mask(self, sent_idxs):
        is_subword_ends = [self.sents[idx].is_subword_end for idx in sent_idxs]
        return torch.tensor(pad_items(is_subword_ends, 0)[0], dtype=torch.bool)

    def _get_length_to_idxs(self, sent_idxs=None):
        if sent_idxs is None:
            sent_idxs = []

        def _to_len(token_ids):
            return len(token_ids)

        return self._get_length_to_idxs_helper(_to_len, sent_idxs)

    def _get_grouped_length_to_idxs(
        self, sent_idxs=None, use_action_len=False, max_length_diff=20
    ):
        if sent_idxs is None:
            sent_idxs = []

        if use_action_len:

            def _get_length(sent):
                return len(sent.action_ids)

        else:

            def _get_length(sent):
                return len(sent.token_ids)

        if len(sent_idxs) == 0:
            sent_idxs = range(len(self.sents))
        length_to_idxs = defaultdict(list)
        group_size = self.group_sentence_size
        sent_idxs_with_len = sorted(
            [(idx, _get_length(self.sents[idx])) for idx in sent_idxs],
            key=lambda x: x[1],
        )

        start_idx = 0
        while start_idx < len(sent_idxs_with_len):
            min_len = sent_idxs_with_len[start_idx][1]
            max_len = sent_idxs_with_len[
                min(start_idx + group_size, len(sent_idxs_with_len) - 1)
            ][1]
            if (
                max_len - min_len < max_length_diff
            ):  # small difference in a group -> register as a group
                end_idx = start_idx + group_size
                group = [idx for idx, length in sent_idxs_with_len[start_idx:end_idx]]
                start_idx += group_size
            else:
                end_idx = start_idx + 1
                while (
                    end_idx < len(sent_idxs_with_len)
                    and sent_idxs_with_len[end_idx][1] - min_len < max_length_diff
                ):
                    end_idx += 1
                group = [idx for idx, length in sent_idxs_with_len[start_idx:end_idx]]
                start_idx = end_idx
            length_to_idxs[_get_length(self.sents[group[-1]])] += group
        return length_to_idxs

    def _get_random_length_to_idxs(self, sent_idxs=None):
        if sent_idxs is None:
            sent_idxs = []

        def _to_len(token_ids):
            return 1  # all sentences belong to the same group

        return self._get_length_to_idxs_helper(_to_len, sent_idxs)

    def _get_length_to_idxs_helper(self, calc_len_func, sent_idxs=None):
        if sent_idxs is None:
            sent_idxs = []
        if len(sent_idxs) == 0:
            sent_idxs = range(len(self.sents))
        length_to_idxs = defaultdict(list)
        for idx in sent_idxs:
            sent = self.sents[idx]
            length_to_idxs[calc_len_func(sent.token_ids)].append(idx)
        return length_to_idxs

    def _pad_action_ids(self, action_ids):
        action_ids, _ = pad_items(action_ids, self.action_dict.padding_idx)
        return action_ids

    def _pad_token_ids(self, token_ids):
        token_ids, _ = pad_items(token_ids, self.vocab.padding_idx)
        return token_ids

    def __len__(self):
        return self.num_batches


class DatasetForAccelerator(torch.utils.data.Dataset):
    def __init__(self, sents):
        self.sents = sents

    def __getitem__(self, index):
        token_ids = self.sents[index].token_ids
        action_ids = self.sents[index].action_ids
        subword_end_mask = self.sents[index].is_subword_end
        max_stack_size = torch.tensor(self.sents[index].max_stack_size)
        return token_ids, action_ids, subword_end_mask, max_stack_size

    def __len__(self):
        return len(self.sents)
