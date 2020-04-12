#include <iostream>
#include <vector>
#include <unordered_map>

#define _HASH_PAIR(x, y) (((x) << 14) | (y))
#define _GET_0(x) ((x) >> 14)
#define _GET_1(x) ((x) & ((1 << 14) - 1))

typedef std::unordered_map<int, std::unordered_map<int, int> > WeightDictType;
typedef std::vector<int> MappingType;

int _hash_pair(int x, int y);

int _get_0(int x);

int _get_1(int x);

int move_gain(MappingType & mapping,
    int node_id,
    int old_id,
    int new_id,
    WeightDictType & weight_dict,
    int match_num);

int swap_gain(MappingType & mapping,
    int node_id1,
    int mapping_id1,
    int node_id2,
    int mapping_id2,
    WeightDictType & weight_dict,
    int match_num);

