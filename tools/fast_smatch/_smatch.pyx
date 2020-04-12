cimport cython
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
import sys
import random


# Error log location
ERROR_LOG = sys.stderr

# Debug log location
DEBUG_LOG = sys.stderr


cdef extern from "_gain.h":
    cdef int _hash_pair(int x, int y)
    cdef int _get_0(int x)
    cdef int _get_1(int x)
    cdef int move_gain(vector[int] & mapping,
                       int node_id, int old_id, int new_id,
                       unordered_map[int, unordered_map[int, int]] & weight_dict, int match_num)
    cdef int swap_gain(vector[int] & mapping,
                       int node_id1, int mapping_id1, int node_id2, int mapping_id2,
                       unordered_map[int, unordered_map[int, int]] & weight_dict, int match_num)


@cython.boundscheck(False)
cdef void smart_init_mapping(const vector[unordered_set[int]] & candidate_mapping,
                             instance1, instance2,
                             vector[int] & result):
    """
    Initialize mapping based on the concept mapping (smart initialization)
    Arguments:
        candidate_mapping: candidate node match list
        instance1: instance triples of AMR 1
        instance2: instance triples of AMR 2
        result:
    Returns:
        initialized node mapping between two AMRs

    """
    random.seed()
    cdef unordered_set[int] matched_dict
    # list to store node indices that have no concept match
    cdef vector[int] no_word_match
    result.clear()
    for i in range(len(candidate_mapping)):
        candidates = candidate_mapping.at(i)
        if len(candidates) == 0:
            # no possible mapping
            result.push_back(-1)
            continue
        # node value in instance triples of AMR 1
        value1 = instance1[i][2]
        for node_index in candidates:
            value2 = instance2[node_index][2]
            # find the first instance triple match in the candidates
            # instance triple match is having the same concept value
            if value1 == value2:
                if matched_dict.count(node_index) == 0:
                    result.push_back(node_index)
                    matched_dict.insert(node_index)
                    break
        if len(result) == i:
            no_word_match.push_back(i)
            result.push_back(-1)
    # if no concept match, generate a random mapping
    for i in no_word_match:
        candidates = [m for m in candidate_mapping.at(i) if matched_dict.count(m) == 0]
        if len(candidates) > 0:
            result[i] = random.choice(candidates)
            matched_dict.insert(result[i])


@cython.boundscheck(False)
cdef void random_init_mapping(const vector[unordered_set[int]] & candidate_mapping,
                              vector[int] & result):
    """
    Generate a random node mapping.
    Args:
        candidate_mapping: candidate_mapping: candidate node match list
        result:
    Returns:
        randomly-generated node mapping between two AMRs

    """
    # if needed, a fixed seed could be passed here to generate same random (to help debugging)
    random.seed()
    cdef unordered_set[int] matched_dict
    result.clear()
    for i in range(len(candidate_mapping)):
        candidates = [m for m in candidate_mapping.at(i) if matched_dict.count(m) == 0]
        if len(candidates) == 0:
            # -1 indicates no possible mapping
            result.push_back(-1)
        else:
            rid = random.choice(candidates)
            result.push_back(rid)
            matched_dict.insert(rid)


@cython.boundscheck(False)
cdef void compute_pool(instance1, attribute1, relation1,
                       instance2, attribute2, relation2,
                       prefix1, prefix2,
                       vector[unordered_set[int]] & candidate_mapping,
                       unordered_map[int, unordered_map[int, int]] & weight_dict):
    """
    compute all possible node mapping candidates and their weights (the triple matching number gain resulting from
    mapping one node in AMR 1 to another node in AMR2)

    Arguments:
        instance1: instance triples of AMR 1
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
        candidate_mapping:
        weight_dict:
    Returns:
      candidate_mapping: a list of candidate nodes.
                       The ith element contains the node indices (in AMR 2) the ith node (in AMR 1) can map to.
                       (resulting in non-zero triple match)
      weight_dict: a dictionary which contains the matching triple number for every pair of node mapping. The key
                   is a node pair. The value is another dictionary. key {-1} is triple match resulting from this node
                   pair alone (instance triples and attribute triples), and other keys are node pairs that can result
                   in relation triple match together with the first node pair.


    """
    candidate_mapping.clear()
    weight_dict.clear()

    cdef int node_pair
    cdef int node_pair1
    cdef int node_pair2
    for i in range(len(instance1)):
        # each candidate mapping is a set of node indices
        candidate_mapping.push_back(unordered_set[int]())
        for j in range(len(instance2)):
            # if both triples are instance triples and have the same valueprint(G.edges(data=True))
            if instance1[i][0].lower() == instance2[j][0].lower() \
                    and instance1[i][2].lower() == instance2[j][2].lower():
                # get node index by stripping the prefix
                node1_index = int(instance1[i][1][len(prefix1):])
                node2_index = int(instance2[j][1][len(prefix2):])
                candidate_mapping[node1_index].insert(node2_index)
                node_pair = _hash_pair(node1_index, node2_index)
                weight_dict[node_pair][-1] += 1
    for i in range(0, len(attribute1)):
        for j in range(0, len(attribute2)):
            # if both attribute relation triple have the same relation name and value
            if attribute1[i][0].lower() == attribute2[j][0].lower() \
                    and attribute1[i][2].lower() == attribute2[j][2].lower():
                node1_index = int(attribute1[i][1][len(prefix1):])
                node2_index = int(attribute2[j][1][len(prefix2):])
                candidate_mapping[node1_index].insert(node2_index)
                node_pair = _hash_pair(node1_index, node2_index)
                weight_dict[node_pair][-1] += 1
    for i in range(0, len(relation1)):
        for j in range(0, len(relation2)):
            # if both relation share the same name
            if relation1[i][0].lower() == relation2[j][0].lower():
                node1_index_amr1 = int(relation1[i][1][len(prefix1):])
                node1_index_amr2 = int(relation2[j][1][len(prefix2):])
                node2_index_amr1 = int(relation1[i][2][len(prefix1):])
                node2_index_amr2 = int(relation2[j][2][len(prefix2):])
                # add mapping between two nodes
                candidate_mapping[node1_index_amr1].insert(node1_index_amr2)
                candidate_mapping[node2_index_amr1].insert(node2_index_amr2)
                node_pair1 = _hash_pair(node1_index_amr1, node1_index_amr2)
                node_pair2 = _hash_pair(node2_index_amr1, node2_index_amr2)
                if node_pair2 != node_pair1:
                    # update weight_dict weight. Note that we need to update both entries for future search
                    # i.e weight_dict[node_pair1][node_pair2]
                    #     weight_dict[node_pair2][node_pair1]
                    if node1_index_amr1 > node2_index_amr1:
                        # swap node_pair1 and node_pair2
                        node_pair1 = _hash_pair(node2_index_amr1, node2_index_amr2)
                        node_pair2 = _hash_pair(node1_index_amr1, node1_index_amr2)
                    if weight_dict.count(node_pair1):
                        weight_dict[node_pair1][node_pair2] += 1
                    else:
                        weight_dict[node_pair1][-1] = 0
                        weight_dict[node_pair1][node_pair2] = 1
                    if weight_dict.count(node_pair2):
                        weight_dict[node_pair2][node_pair1] += 1
                    else:
                        weight_dict[node_pair2][-1] = 0
                        weight_dict[node_pair2][node_pair1] = 1
                else:
                    # two node pairs are the same. So we only update weight_dict once.
                    # this generally should not happen.
                    weight_dict[node_pair1][-1] += 1


@cython.boundscheck(False)
cdef int compute_match(const vector[int] & mapping,
                       unordered_map[int, unordered_map[int, int]] & weight_dict,
                       verbose=False):
    """
    Given a node mapping, compute match number based on weight_dict.
    Args:
    mappings: a list of node index in AMR 2. The ith element (value j) means node i in AMR 1 maps to node j in AMR 2.
    Returns:
    matching triple number
    Complexity: O(m*n) , m is the node number of AMR 1, n is the node number of AMR 2

    """
    # If this mapping has been investigated before, retrieve the value instead of re-computing.
    if verbose:
        print >> DEBUG_LOG, "Computing match for mapping"
        print >> DEBUG_LOG, mapping
    cdef match_num = 0
    # i is node index in AMR 1, m is node index in AMR 2
    for i, m in enumerate(mapping):
        if m == -1:
            # no node maps to this node
            continue
        # node i in AMR 1 maps to node m in AMR 2
        current_node_pair = _hash_pair(i, m)
        if weight_dict.count(current_node_pair) == 0:
            continue
        if verbose:
            print >> DEBUG_LOG, "node_pair", current_node_pair
        for key in weight_dict[current_node_pair]:
            if key.first == -1:
                # matching triple resulting from instance/attribute triples
                match_num += key.second
                if verbose:
                    print >> DEBUG_LOG, "instance/attribute match", key.second
                continue
            # only consider node index larger than i to avoid duplicates
            # as we store both weight_dict[node_pair1][node_pair2] and
            #     weight_dict[node_pair2][node_pair1] for a relation
            first = _get_0(key.first)
            if first < i:
                continue
            elif mapping[first] == _get_1(key.first):
                match_num += key.second
                if verbose:
                    print >> DEBUG_LOG, "relation match with", key, key.second
    if verbose:
        print >> DEBUG_LOG, "match computing complete, result:", match_num
    return match_num


@cython.boundscheck(False)
cdef int get_best_gain(vector[int] & mapping,
                       vector[unordered_set[int]] & candidate_mappings,
                       unordered_map[int, unordered_map[int, int]] & weight_dict,
                       int instance_len,
                       int cur_match_num,
                       vector[int] & cur_mapping, verbose=False):
    """
    Hill-climbing method to return the best gain swap/move can get
    Arguments:
    mapping: current node mapping
    candidate_mappings: the candidates mapping list
    weight_dict: the weight dictionary
    instance_len: the number of the nodes in AMR 2
    cur_match_num: current triple match number
    Returns:
    the best gain we can get via swap/move operation

    """
    cdef int largest_gain = 0
    cdef int mv_gain = 0
    cdef int sw_gain = 0
    cur_mapping.clear()
    # True: using swap; False: using move
    cdef bint use_swap = 1
    # the node to be moved/swapped
    cdef int node1 = -1
    # store the other node affected. In swap, this other node is the node swapping with node1. In move, this other
    # node is the node node1 will move to.
    cdef int node2 = -1
    # unmatched nodes in AMR 2
    unmatched = set(range(0, instance_len))
    # exclude nodes in current mapping
    # get unmatched nodes
    for nid in mapping:
        if nid in unmatched:
            unmatched.remove(nid)
    for i, nid in enumerate(mapping):
        # current node i in AMR 1 maps to node nid in AMR 2
        for nm in unmatched:
            if candidate_mappings[i].count(nm):
                # remap i to another unmatched node (move)
                # (i, m) -> (i, nm)
                if verbose:
                    print >> DEBUG_LOG, "Remap node", i, "from ", nid, "to", nm
                mv_gain = move_gain(mapping, i, nid, nm, weight_dict, cur_match_num)
                if verbose:
                    print >> DEBUG_LOG, "Move gain:", mv_gain
                    new_mapping = mapping[:]
                    new_mapping[i] = nm
                    new_match_num = compute_match(new_mapping, weight_dict)
                    if new_match_num != cur_match_num + mv_gain:
                        print >> ERROR_LOG, mapping, new_mapping
                        print >> ERROR_LOG, "Inconsistency in computing: move gain", cur_match_num, mv_gain, \
                            new_match_num
                if mv_gain > largest_gain:
                    largest_gain = mv_gain
                    node1 = i
                    node2 = nm
                    use_swap = 0
    # compute swap gain
    for i, m in enumerate(mapping):
        for j in range(i+1, len(mapping)):
            m2 = mapping[j]
            # swap operation (i, m) (j, m2) -> (i, m2) (j, m)
            # j starts from i+1, to avoid duplicate swap
            if verbose:
                print >> DEBUG_LOG, "Swap node", i, "and", j
                print >> DEBUG_LOG, "Before swapping:", i, "-", m, ",", j, "-", m2
                print >> DEBUG_LOG, mapping
                print >> DEBUG_LOG, "After swapping:", i, "-", m2, ",", j, "-", m
            sw_gain = swap_gain(mapping, i, m, j, m2, weight_dict, cur_match_num)
            if verbose:
                print >> DEBUG_LOG, "Swap gain:", sw_gain
                new_mapping = mapping[:]
                new_mapping[i] = m2
                new_mapping[j] = m
                print >> DEBUG_LOG, new_mapping
                new_match_num = compute_match(new_mapping, weight_dict)
                if new_match_num != cur_match_num + sw_gain:
                    # print >> ERROR_LOG, match, new_match
                    print >> ERROR_LOG, "Inconsistency in computing: swap gain", cur_match_num, sw_gain, new_match_num
            if sw_gain > largest_gain:
                largest_gain = sw_gain
                node1 = i
                node2 = j
                use_swap = 1
    # generate a new mapping based on swap/move
    cur_mapping.assign(mapping.begin(), mapping.end())
    if node1 >= 0:
        if use_swap:
            if verbose:
                print >> DEBUG_LOG, "Use swap gain"
            temp = cur_mapping[node1]
            cur_mapping[node1] = cur_mapping[node2]
            cur_mapping[node2] = temp
        else:
            if verbose:
                print >> DEBUG_LOG, "Use move gain"
            cur_mapping[node1] = node2
    else:
        if verbose:
            print >> DEBUG_LOG, "no move/swap gain found"
    if verbose:
        print >> DEBUG_LOG, "Original mapping", mapping
        print >> DEBUG_LOG, "Current mapping", cur_mapping
    return largest_gain


@cython.boundscheck(False)
cdef void remove_zero_alignments(vector[int] & mapping,
                                 unordered_map[int, unordered_map[int, int]] & weight_dict):
    for i, m in enumerate(mapping):
        match_num = 0
        if m == -1:
            # no node maps to this node
            continue
        # node i in AMR 1 maps to node m in AMR 2
        current_node_pair = _hash_pair(i, m)
        if weight_dict.count(current_node_pair) == 0:
            continue
        for key in weight_dict[current_node_pair]:
            if key.first == -1:
                # matching triple resulting from instance/attribute triples
                match_num += key.second
            # only consider node index larger than i to avoid duplicates
            # as we store both weight_dict[node_pair1][node_pair2] and
            #     weight_dict[node_pair2][node_pair1] for a relation
            elif _get_0(key.first) < i:
                continue
            elif mapping[_get_0(key.first)] == _get_1(key.first):
                match_num += key.second
        if match_num == 0:
            mapping[i] = -1


@cython.boundscheck(False)
cpdef get_best_match(instance1, attribute1, relation1,
                     instance2, attribute2, relation2,
                     prefix1, prefix2, iteration_num=5, verbose=False):
    """
    Get the highest triple match number between two sets of triples via hill-climbing.
    Arguments:
        instance1: instance triples of AMR 1 ("instance", node name, node value)
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2 ("instance", node name, node value)
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name)
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
        iteration_num:
        verbose:
    Returns:
        best_match: the node mapping that results in the highest triple matching number
        best_match_num: the highest triple matching number

    """
    # Compute candidate pool - all possible node match candidates.
    # In the hill-climbing, we only consider candidate in this pool to save computing time.
    # weight_dict is a dictionary that maps a pair of node
    cdef vector[unordered_set[int]] candidate_mappings
    cdef unordered_map[int, unordered_map[int, int]] weight_dict

    compute_pool(instance1, attribute1, relation1, instance2, attribute2, relation2,
                 prefix1, prefix2, candidate_mappings, weight_dict)

    if verbose:
        print >> DEBUG_LOG, "Candidate mappings:"
        print >> DEBUG_LOG, candidate_mappings
        print >> DEBUG_LOG, "Weight dictionary"
        print >> DEBUG_LOG, weight_dict
    cdef int best_match_num = 0
    cdef int match_num = 0
    # initialize best match mapping
    # the ith entry is the node index in AMR 2 which maps to the ith node in AMR 1
    cdef vector[int] best_mapping = vector[int](len(instance1), -1)
    cdef vector[int] cur_mapping
    cdef vector[int] new_mapping
    for i in range(iteration_num):
        if verbose:
            print >> DEBUG_LOG, "Iteration", i
        if i == 0:
            # smart initialization used for the first round
            smart_init_mapping(candidate_mappings, instance1, instance2, cur_mapping)
        else:
            # random initialization for the other round
            random_init_mapping(candidate_mappings, cur_mapping)
        # compute current triple match number
        match_num = compute_match(cur_mapping, weight_dict)
        if verbose:
            print >> DEBUG_LOG, "Node mapping at start", cur_mapping
            print >> DEBUG_LOG, "Triple match number at start:", match_num
        while True:
            # get best gain
            gain = get_best_gain(cur_mapping, candidate_mappings, weight_dict,
                                 len(instance2), match_num, new_mapping)
            if verbose:
                print >> DEBUG_LOG, "Gain after the hill-climbing", gain
            # hill-climbing until there will be no gain for new node mapping
            if gain <= 0:
                break
            # otherwise update match_num and mapping
            match_num += gain
            cur_mapping = new_mapping
            if verbose:
                print >> DEBUG_LOG, "Update triple match number to:", match_num
                print >> DEBUG_LOG, "Current mapping:", cur_mapping
        if match_num > best_match_num:
            best_mapping = cur_mapping
            best_match_num = match_num
    remove_zero_alignments(best_mapping, weight_dict)
    return best_mapping, best_match_num


def clear_match_triple_dict():
    # dummy for api
    pass


def compute_f(match_num, test_num, gold_num, verbose=False):
    """
    Compute the f-score based on the matching triple number,
                                 triple number of AMR set 1,
                                 triple number of AMR set 2
    Args:
        match_num: matching triple number
        test_num:  triple number of AMR 1 (test file)
        gold_num:  triple number of AMR 2 (gold file)
        verbose
    Returns:
        precision: match_num/test_num
        recall: match_num/gold_num
        f_score: 2*precision*recall/(precision+recall)
    """
    if test_num == 0 or gold_num == 0:
        return 0.00, 0.00, 0.00
    precision = float(match_num) / float(test_num)
    recall = float(match_num) / float(gold_num)
    if (precision + recall) != 0:
        f_score = 2 * precision * recall / (precision + recall)
        if verbose:
            print >> DEBUG_LOG, "F-score:", f_score
        return precision, recall, f_score
    else:
        if verbose:
            print >> DEBUG_LOG, "F-score:", "0.0"
        return precision, recall, 0.00
