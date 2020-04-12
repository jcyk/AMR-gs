import penman
import re
import networkx as nx
import numpy as np

from parser.AMRGraph import _is_attr_form, need_an_instance

class PostProcessor(object):
    def __init__(self, rel_vocab):
        self.amr = penman.AMRCodec()
        self.rel_vocab = rel_vocab

    def to_triple(self, res_concept, res_relation):
        """ res_concept: list of strings
            res_relation: list of (dep:int, head:int, arc_prob:float, rel_prob:list(vocab))
        """
        ret = []
        names = []
        for i, c in enumerate(res_concept):
            if need_an_instance(c):
                name = 'c' + str(i)
                ret.append((name, 'instance', c))
            else:
                if c.endswith('_'):
                    name = '"'+c[:-1]+'"'
                else:
                    name = c
                name = name + '@attr%d@'%i
            names.append(name)

        grouped_relation = dict()
        for i, j, p, r in res_relation:
            r = self.rel_vocab.idx2token(np.argmax(np.array(r)))
            grouped_relation[i] = grouped_relation.get(i, []) + [(j, p, r)]
        for i, c in enumerate(res_concept):
            if i ==0:
                continue
            max_p, max_j, max_r = 0., 0, None
            for j, p, r in grouped_relation[i]:
                assert j < i
                if _is_attr_form(res_concept[j]):
                    continue
                if p >=0.5:
                    if not _is_attr_form(res_concept[i]):
                        if r.endswith('_reverse_'):
                            ret.append((names[i], r[:-9], names[j]))
                        else:
                            ret.append((names[j], r, names[i]))
                if p > max_p:
                    max_p = p
                    max_j = j
                    max_r = r

            if max_p < 0.5 or _is_attr_form(res_concept[i]):
                if max_r.endswith('_reverse_'):
                    ret.append((names[i], max_r[:-9], names[max_j]))
                else:
                    ret.append((names[max_j], max_r, names[i]))
        return ret

    def get_string(self, x):
        return self.amr.encode(penman.Graph(x), top=x[0][0])
    
    def postprocess(self, concept, relation):
        mstr = self.get_string(self.to_triple(concept, relation))
        return re.sub(r'@attr\d+@', '', mstr)
