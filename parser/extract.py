#!/usr/bin/env python
# coding: utf-8
from collections import Counter
import json, re

from parser.amr import AMR
from parser.AMRGraph import AMRGraph, number_regexp

class AMRIO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [ le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                    graph_line = AMR.get_amr_line(f)
                    amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(amr)
                    yield tokens, lemmas, pos_tags, ner_tags, myamr

class LexicalMap(object):


    # build our lexical mapping (from token/lemma to concept), useful for copy mechanism.
    def __init__(self):
        pass

    #cp_seq, mp_seq, token2idx, idx2token = lex_map.get(lemma, token, vocabs['predictable_concept'])
    def get_concepts(self, lem, tok, vocab=None):
        cp_seq, mp_seq = [], []
        new_tokens = set()
        for le, to in zip(lem, tok):
            cp_seq.append(le+'_')
            mp_seq.append(le)

        if vocab is None:
            return cp_seq, mp_seq

        for cp, mp in zip(cp_seq, mp_seq):
            if vocab.token2idx(cp) == vocab.unk_idx:
                new_tokens.add(cp)
            if vocab.token2idx(mp) == vocab.unk_idx:
                new_tokens.add(mp)
        nxt = vocab.size
        token2idx, idx2token = dict(), dict()
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt += 1
        return cp_seq, mp_seq, token2idx, idx2token


def read_file(filename):
    # read preprocessed amr file
    token, lemma, pos, ner, amrs = [], [], [], [], []
    for _tok, _lem, _pos, _ner, _myamr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        pos.append(_pos)
        ner.append(_ner)
        amrs.append(_myamr)
    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, pos, ner

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    amrs, token, lemma, pos, ner = read_file(args.train_data)
    lexical_map = LexicalMap()

    # collect concepts and relations
    conc = []
    rel = []
    predictable_conc = []
    for i in range(10):
        # run 10 times random sort to get the priorities of different types of edges
        for amr, lem, tok in zip(amrs, lemma, token):
            concept, edge, not_ok = amr.root_centered_sort()
            lexical_concepts = set()
            cp_seq, mp_seq = lexical_map.get_concepts(lem, tok)
            for lc, lm in zip(cp_seq, mp_seq):
                lexical_concepts.add(lc)
                lexical_concepts.add(lm)

            if i == 0:
                predictable_conc.append([ c for c in concept if c not in lexical_concepts])
                conc.append(concept)
            rel.append([e[-1] for e in edge])

    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    pos_vocab = make_vocab(pos)
    ner_vocab = make_vocab(ner)
    conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

    predictable_conc_vocab = make_vocab(predictable_conc)
    num_predictable_conc = sum(len(x) for x in predictable_conc)
    num_conc = sum(len(x) for x in conc)
    print ('predictable concept coverage', num_predictable_conc, num_conc, num_predictable_conc/num_conc)
    rel_vocab = make_vocab(rel)

    print ('make vocabularies')
    write_vocab(token_vocab, 'tok_vocab')
    write_vocab(token_char_vocab, 'word_char_vocab')
    write_vocab(lemma_vocab, 'lem_vocab')
    write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(pos_vocab, 'pos_vocab')
    write_vocab(ner_vocab, 'ner_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(predictable_conc_vocab, 'predictable_concept_vocab')
    write_vocab(rel_vocab, 'rel_vocab')
