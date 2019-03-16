# mostly from https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f/
# co-occurrence is new

"""Module that handles all augmentation."""
import random
import sys
from collections import defaultdict
from typing import Tuple, Dict, Set, Optional, List

from domains import DomainDataString
from grammar import Grammar
from utils import stopwords

END_OF_SENTENCE = '</s>'


class Augmenter(object):
    def __init__(self, domain, dataset, aug_types):
        self.domain = domain
        self.dataset = dataset
        self.dataset_set = set(dataset)
        self.setup_grammar(aug_types)

    def setup_grammar(self, aug_types):
        grammar = Grammar(self.dataset)
        for aug_type in aug_types:
            if aug_type == 'entity':
                grammar = self.induce_entity_grammar(grammar)
            elif aug_type == 'nesting':
                grammar = self.induce_nesting_grammar(grammar)
            elif aug_type.startswith('concat'):
                concat_num = int(aug_type[6:])
                grammar = self.induce_concat_grammar(grammar, concat_num)
        self.grammar = grammar

    def splice(self, s, swaps):
        # Process s from right to left
        swaps.sort(key=lambda x: x[0], reverse=True)

        cur_left = len(s)
        new_s = s
        for span, rep in swaps:
            # Ensure disjoint
            if span[1] > cur_left:
                print(s, file=sys.stderr)  # >> sys.stderr, s
                print(swaps, file=sys.stderr)  # print >> sys.stderr, swaps
                raise ValueError('Non-disjoint spans detected')
            new_s = new_s[:span[0]] + rep + new_s[span[1]:]
            cur_left = span[0]
        return new_s

    def induce_entity_grammar(self, start_grammar):
        """Induce an entity-swapping grammar.

        Get the entities from the original dataset.
        Get the places to put holes from start_grammar.
        """
        new_grammar = Grammar()

        # Entity rules
        for x, y in self.dataset:
            alignments = self.domain.get_entity_alignments(x, y)
            for cat, x_span, y_span in alignments:
                x_str = x[x_span[0]:x_span[1]]
                y_str = y[y_span[0]:y_span[1]]
                new_grammar.add_rule(cat, x_str, y_str)

        # Root/template rules
        for cat, x_str, y_str in start_grammar.rule_list:
            # Anchor on single mention in x--allow one-to-many x-to-y mapping
            alignments = self.domain.get_entity_alignments(x_str, y_str)
            x_swaps = list(set(
                [(x_span, '%s_%d' % (inner_cat, x_span[0]))
                 for i, (inner_cat, x_span, y_span) in enumerate(alignments)]))
            x_new = self.splice(x_str, x_swaps)
            y_swaps = [(y_span, '%s_%d' % (inner_cat, x_span[0]))
                       for i, (inner_cat, x_span, y_span) in enumerate(alignments)]
            y_new = self.splice(y_str, y_swaps)
            new_grammar.add_rule(cat, x_new, y_new)

        # new_grammar.print_self()
        return new_grammar

    def induce_nesting_grammar(self, start_grammar):
        """Induce an entity-swapping grammar.

        Get everything from the start_grammar.
        """
        new_grammar = Grammar()
        for cat, x_str, y_str in start_grammar.rule_list:
            alignments, productions = self.domain.get_nesting_alignments(x_str, y_str)

            # Replacements
            for cat_p, x_p, y_p in productions:
                new_grammar.add_rule(cat_p, x_p, y_p)

            # Templates
            x_swaps = list(set(
                [(x_span, '%s_%d' % (inner_cat, x_span[0]))
                 for i, (inner_cat, x_span, y_span) in enumerate(alignments)]))
            x_new = self.splice(x_str, x_swaps)
            y_swaps = [(y_span, '%s_%d' % (inner_cat, x_span[0]))
                       for i, (inner_cat, x_span, y_span) in enumerate(alignments)]
            y_new = self.splice(y_str, y_swaps)
            new_grammar.add_rule(cat, x_new, y_new)
        new_grammar.print_self()
        return new_grammar

    def induce_concat_grammar(self, start_grammar, concat_num):
        new_grammar = Grammar()

        for cat, x_str, y_str in start_grammar.rule_list:
            if cat == start_grammar.ROOT:
                new_grammar.add_rule('$sentence', x_str, y_str)
            else:
                new_grammar.add_rule(cat, x_str, y_str)
        root_str = (' %s ' % END_OF_SENTENCE).join(
            '$sentence_%d' % i for i in range(concat_num))
        new_grammar.add_rule(new_grammar.ROOT, root_str, root_str)
        # new_grammar.print_self()
        return new_grammar

    def sample(self, num):
        aug_data = []
        while len(aug_data) < num:
            x, y = self.grammar.sample()
            if (x, y) in self.dataset_set:
                continue
            aug_data.append((x, y))
        return aug_data


class CoOccurrence:
    def __init__(self, src_tgt: DomainDataString):
        # augment with this probability
        self.aug_prob = 0.5

        self.src_tgt = src_tgt
        self.lexicon = self._generate_lexicon(self.src_tgt)

    @staticmethod
    def _generate_lexicon(src_tgt: DomainDataString) -> Dict[str, Set[str]]:
        """
        Build a mapping between co-occurring tokens.
        Only considering source sentences of the same length with a single token difference.
        For example for the following two sentences:
        - what is the capital of wyoming?
        - what is the capital of alaska?
        we will generate a mapping from alaska -> wyoming and another one from wyoming -> alaska in the lexicon
        the lexicon would look like this if it only had these two records:
        {
            'wyoming': {'alaska'},
            'alaska': {'wyoming'}
        }
        as more co-occurring tokens are found they're added to each set.

        :param src_tgt: domain data e.g. [('what is the cpaital of...', ('_answer ...'), ...]
        :return: lexicon
        """
        srcs = [set(src.split()) for src, _ in src_tgt]

        co_occur_map = defaultdict(set)

        co_occur = [
            (list(b.difference(c))[0], list(c.difference(b))[0])
            for b in srcs
            for c in srcs
            if (b != c
                and len(b.intersection(c)) == len(b) - 1
                and len(b) == len(c)
                and list(b.difference(c))[0] not in stopwords
                and list(c.difference(b))[0] not in stopwords)
        ]

        for a, b in co_occur:
            co_occur_map[a].add(b)
            co_occur_map[b].add(a)

        return co_occur_map

    def _get_src_mapping(self, xs: List[str]) -> Dict[str, str]:
        """
        Genereates the mapping of the tokens to be replaced, randomly selected from the available ones in the lexicon.
        :param xs: list of tokens in a source sentence, e.g. ['what', 'is', 'the', 'capital', ...]
        :return: mapping, e.g. {'wyoming': 'alabama', 'indiana': 'california', ...}
        """
        return {
            x: random.choice(list(self.lexicon[x]))
            for x in xs
            if x in self.lexicon and random.random() <= self.aug_prob
        }

    def _sample_item(self, x_str: str, y_str: str) -> Tuple[str, str]:
        """
        Replace the tokens given the mapping in both the natural language utterance and the logical form.
        :param x_str: natural language utterance, e.g. 'what is the capital of ...'
        :param y_str: logical form, e.g. '_answer _capital ( V0 ) ...'
        :return: augmented pair
        """
        x_lst, y_lst = x_str.split(), y_str.split()
        mapping = self._get_src_mapping(x_lst)
        xs_aug = ' '.join([mapping[x] if x in mapping else x for x in x_lst])
        ys_aug = ' '.join([mapping[y] if y in mapping else y for y in y_lst])
        return xs_aug, ys_aug

    def __call__(self, src_tgt: DomainDataString, n: Optional[int] = None) -> DomainDataString:
        """

        :param src_tgt: dataset to augment
        :param n: number of random pairs to augment where n <= len(src_tgt), if None then augment all
        :return: augmented pairs
        """
        if n:
            assert n > 0, f"n must be positive, it's {n}"
            src_tgt = random.sample(src_tgt, n)
        # return src_tgt
        return [self._sample_item(x_str, y_str) for x_str, y_str in src_tgt]
