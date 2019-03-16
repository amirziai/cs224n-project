# mostly from https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f/
# co-occurrence is new

"""Module that handles all augmentation."""
import random
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Set

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


CoOccurrenceData = List[Tuple[str, str]]


class CoOccurrence:
    def __init__(self, src_tgt: CoOccurrenceData):
        self.window_size = 3
        self.token_size_threshold = 3
        self.support = 0.3
        self.support_abs = 2
        self.aug_prob = 0.5

        self.src_tgt = src_tgt
        self.lexicon_src_tgt = self._generate_lexicon(self.src_tgt, is_reverse=False)
        self.lexicon_tgt_src = self._generate_lexicon([(y, x) for x, y in self.src_tgt], is_reverse=True)

    def _generate_lexicon(self, src_tgt: List[Tuple[str, str]], is_reverse: bool = False) -> Dict[str, Set[str]]:
        cnt = defaultdict(lambda: defaultdict(int))
        cnt_tgt = defaultdict(int)

        # _answer appears everywhere for target -> source
        stop_words = set(stopwords).union({'_answer'} if is_reverse else set())

        for src, tgt in src_tgt:
            src_tokens = src.split()
            tgt_tokens = tgt.split()

            for tgt_token in tgt_tokens:
                cnt_tgt[tgt_token] += 1

            for i, src_token in enumerate(src_tokens):
                if len(src_token) >= self.token_size_threshold and src_token not in stop_words:
                    for j in range(-self.window_size, self.window_size + 1):
                        if (0 <= i + j < len(tgt_tokens) and len(tgt_tokens[i + j]) >= self.token_size_threshold and
                                tgt_tokens[i + j] not in stop_words):
                            cnt[tgt_tokens[i + j]][src_token] += 1

        # set of src words that positionally (within a window) co-occur with some min support with the target
        x = {
            tgt: {src: cnt[tgt][src] for src in cnt[tgt] if
                  cnt[tgt][src] >= self.support * cnt_tgt[tgt] and cnt[tgt][src] >= self.support_abs}
            for tgt in cnt
            if
            len({src for src in cnt[tgt] if
                 cnt[tgt][src] >= self.support * cnt_tgt[tgt] and cnt[tgt][src] >= self.support_abs}) >= 2
        }

        # lookup for co-occurring tokens
        out = defaultdict(set)
        for val_dict in x.values():
            val_vals = val_dict.keys()
            for val in val_vals:
                for val_map_to in val_vals:
                    if val_map_to != val:
                        out[val].add(val_map_to)

        return out

    def _aug(self, token: str, src: bool) -> str:
        lexicon = self.lexicon_src_tgt if src else self.lexicon_tgt_src
        if token in lexicon:
            do_aug = random.random() <= self.aug_prob
            return random.choice(list(lexicon[token])) if do_aug else token
        else:
            return token

    def _sample_item(self, x_str: str, y_str: str) -> Tuple[str, str]:
        x_lst, y_lst = x_str.split(), y_str.split()
        xs_aug = ' '.join([self._aug(x, src=True) for x in x_lst])
        ys_aug = ' '.join([self._aug(y, src=False) for y in y_lst])
        return xs_aug, ys_aug

    def sample(self, n) -> CoOccurrenceData:
        src_tgt_n = random.sample(self.src_tgt, n)
        return [self._sample_item(x_str, y_str) for x_str, y_str in src_tgt_n]

# def main():
#     """Print augmented data to stdout."""
#     if len(sys.argv) < 5:
#         print >> sys.stderr, 'Usage: %s [file] [domain] [aug-type] [num]' % sys.argv[0]
#         sys.exit(1)
#     fname, domain_name, aug_type_str, num = sys.argv[1:5]
#     num = int(num)
#     aug_types = aug_type_str.split('+')
#     data = []
#     domain = domains.new(domain_name)
#     with open(fname) as f:
#         for line in f:
#             x, y = line.strip().split('\t')
#             y = domain.preprocess_lf(y)
#             data.append((x, y))
#     augmenter = Augmenter(domain, data, aug_types)
#     aug_data = augmenter.sample(num)
#     for ex in aug_data:
#         print
#         '\t'.join(ex)
#
#
# if __name__ == '__main__':
#     main()
