#!/usr/bin/env python
from __future__ import print_function
from amr import AMR
try:
    from _smatch import get_best_match, compute_f, clear_match_triple_dict
except:
    import sys
    print('WARN: use slow version of smatch api.', file=sys.stderr)
    from smatch import get_best_match, compute_f, clear_match_triple_dict


def _smatch(cur_amr1, cur_amr2, n_iter):
    clear_match_triple_dict()

    amr1 = AMR.parse_AMR_line(cur_amr1)
    amr2 = AMR.parse_AMR_line(cur_amr2)
    prefix1 = "a"
    prefix2 = "b"

    amr1.rename_node(prefix1)
    amr2.rename_node(prefix2)
    instance1, attributes1, relation1 = amr1.get_triples()
    instance2, attributes2, relation2 = amr2.get_triples()

    best_mapping, best_match_num = get_best_match(instance1, attributes1, relation1,
                                                  instance2, attributes2, relation2,
                                                  prefix1, prefix2)

    test_triple_num = len(instance1) + len(attributes1) + len(relation1)
    gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
    return best_match_num, test_triple_num, gold_triple_num


def smatch(cur_amr1, cur_amr2, n_iter=5):
    best_match_num, test_triple_num, gold_triple_num = _smatch(cur_amr1, cur_amr2, n_iter)
    precision, recall, best_f_score = compute_f(best_match_num, test_triple_num, gold_triple_num)
    return best_f_score


class SmatchScorer(object):
    def __init__(self, n_iter=5):
        self.total_match_num = 0
        self.total_test_num = 0
        self.total_gold_num = 0
        self.last_match_num = 0
        self.last_test_num = 0
        self.last_gold_num = 0
        self.n_iter = n_iter

    def update(self, cur_amr1, cur_amr2):
        best_match_num, test_triple_num, gold_triple_num = _smatch(cur_amr1, cur_amr2, self.n_iter)
        self.last_match_num = best_match_num
        self.last_test_num = test_triple_num
        self.last_gold_num = gold_triple_num

        self.total_match_num += best_match_num
        self.total_test_num += test_triple_num
        self.total_gold_num += gold_triple_num

    def f_score(self):
        return compute_f(self.total_match_num, self.total_test_num, self.total_gold_num)[2]

    def last_f_score(self):
        return compute_f(self.last_match_num, self.last_test_num, self.last_gold_num)[2]

    def reset(self):
        self.total_match_num = 0
        self.total_test_num = 0
        self.total_gold_num = 0
