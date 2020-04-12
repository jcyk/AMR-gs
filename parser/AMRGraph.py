# encoding=utf8
import re
import random

number_regexp = re.compile(r'^-?(\d)+(\.\d+)?$')
abstract_regexp0 = re.compile(r'^([A-Z]+_)+\d+$')
abstract_regexp1 = re.compile(r'^\d0*$')
discard_regexp = re.compile(r'^n(\d+)?$')

attr_value_set = set(['-', '+', 'interrogative', 'imperative', 'expressive'])

def _is_attr_form(x):
    return (x in attr_value_set or x.endswith('_') or number_regexp.match(x) is not None)
def _is_abs_form(x):
    return (abstract_regexp0.match(x) is not None or abstract_regexp1.match(x) is not None)
def is_attr_or_abs_form(x):
    return _is_attr_form(x) or _is_abs_form(x)
def need_an_instance(x):
    return (not _is_attr_form(x) or (abstract_regexp0.match(x) is not None))

class AMRGraph(object):

    def __init__(self, smatch_amr):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple = smatch_amr.get_triples()
        self.root = smatch_amr.root
        self.nodes = set()
        self.edges = dict()
        self.reversed_edges = dict()
        self.undirected_edges = dict()
        self.name2concept = dict()


        # will do some adjustments
        self.abstract_concepts = dict()
        for _, name, concept in instance_triple:
            if is_attr_or_abs_form(concept):
                if _is_abs_form(concept):
                    self.abstract_concepts[name] = concept
                else:
                    print ('bad concept', _, name, concept)
            self.name2concept[name] = concept
            self.nodes.add(name)
        for rel, concept, value in attribute_triple:
            if rel == 'TOP':
                continue
            # discard some empty names
            if rel == 'name' and discard_regexp.match(value):
                continue
            # abstract concept can't have an attribute
            if concept in self.abstract_concepts:
                print (rel, self.abstract_concepts[concept], value, "abstract concept cannot have an attribute")
                continue
            name = "%s_attr_%d"%(value, len(self.name2concept))
            if not _is_attr_form(value):
                if _is_abs_form(value):
                    self.abstract_concepts[name] = value
                else:
                    print ('bad attribute', rel, concept, value)
                    continue
            self.name2concept[name] = value
            self._add_edge(rel, concept, name)
        for rel, head, tail in relation_triple:
            self._add_edge(rel, head, tail)

        # lower concept
        for name in self.name2concept:
            v = self.name2concept[name]
            if not _is_abs_form(v):
                v = v.lower()
            self.name2concept[name] = v

    def __len__(self):
        return len(self.name2concept)

    def _add_edge(self, rel, src, des):
        self.nodes.add(src)
        self.nodes.add(des)
        self.edges[src] = self.edges.get(src, []) + [(rel, des)]
        self.reversed_edges[des] = self.reversed_edges.get(des, []) + [(rel, src)]
        self.undirected_edges[src] = self.undirected_edges.get(src, []) + [(rel, des)]
        self.undirected_edges[des] = self.undirected_edges.get(des, []) + [(rel + '_reverse_', src)]

    def root_centered_sort(self, rel_order=None):
        queue = [self.root]
        visited = set(queue)
        step = 0
        while len(queue) > step:
            src = queue[step]
            step += 1
            if src not in self.undirected_edges:
                continue

            random.shuffle(self.undirected_edges[src])
            if rel_order is not None:
                # Do some random thing here for performance enhancement
                if random.random() < 0.5:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]) if (x[0].startswith('snt') or x[0].startswith('op') ) else -1)
                else:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]))
            for rel, des in self.undirected_edges[src]:
                if des in visited:
                    continue
                else:
                    queue.append(des)
                    visited.add(des)
        not_connected = len(queue) != len(self.nodes)
        assert (not not_connected)
        name2pos = dict(zip(queue, range(len(queue))))

        visited = set()
        edge = []
        for x in queue:
            if x not in self.undirected_edges:
                continue
            for r, y in self.undirected_edges[x]:
                if y in visited:
                    r = r[:-9] if r.endswith('_reverse_') else r+'_reverse_'
                    edge.append((name2pos[x], name2pos[y], r)) # x -> y: r
            visited.add(x)
        return [self.name2concept[x] for x in queue], edge, not_connected
