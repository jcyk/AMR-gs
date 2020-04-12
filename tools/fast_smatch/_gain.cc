#include "_gain.h"

int _hash_pair(int x, int y) {
  return _HASH_PAIR(x, y);
}

int _get_0(int x) {
  return _GET_0(x);
}

int _get_1(int x) {
  return _GET_1(x);
}


int move_gain(MappingType & mapping,
    int node_id,
    int old_id,
    int new_id,
    WeightDictType & weight_dict,
    int match_num) {
  int new_mapping = _HASH_PAIR(node_id, new_id);
  int old_mapping = _HASH_PAIR(node_id, old_id);
  int saved_id = mapping[node_id];

  mapping[node_id] = new_id;
  int gain = 0;
  WeightDictType::const_iterator entry = weight_dict.find(new_mapping);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain += key->second;
      } else if (mapping[_GET_0(key->first)] == _GET_1(key->first)) {
        gain += key->second;
      }
    }
  }

  mapping[node_id] = saved_id;
  entry = weight_dict.find(old_mapping);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain -= key->second;
      } else if (mapping[_GET_0(key->first)] == _GET_1(key->first)) {
        gain -= key->second;
      }
    }
  }
  return gain;
}

int swap_gain(MappingType & mapping,
    int node_id1,
    int mapping_id1,
    int node_id2,
    int mapping_id2,
    WeightDictType & weight_dict,
    int match_num) {
  int saved_id1 = mapping[node_id1];
  int saved_id2 = mapping[node_id2];
  int gain = 0;

  int new_mapping1 = _HASH_PAIR(node_id1, mapping_id2);
  int new_mapping2 = _HASH_PAIR(node_id2, mapping_id1);
  int old_mapping1 = _HASH_PAIR(node_id1, mapping_id1);
  int old_mapping2 = _HASH_PAIR(node_id2, mapping_id2);

  if (node_id1 > node_id2) {
    std::swap(new_mapping1, new_mapping2);
    std::swap(old_mapping1, old_mapping2);
  }

  mapping[node_id1] = mapping_id2;
  mapping[node_id2] = mapping_id1;

  WeightDictType::const_iterator entry = weight_dict.find(new_mapping1);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain += key->second;
      } else if (mapping[_GET_0(key->first)] == _GET_1(key->first)) {
        gain += key->second;
      }
    }
  }

  entry = weight_dict.find(new_mapping2);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain += key->second;
        continue;
      }
      int first = _GET_0(key->first);
      if (first != node_id1 && mapping[first] == _GET_1(key->first)) {
        gain += key->second;
      }
    }
  }

  mapping[node_id1] = saved_id1;
  mapping[node_id2] = saved_id2;

  entry = weight_dict.find(old_mapping1);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain -= key->second;
      } else if (mapping[_GET_0(key->first)] == _GET_1(key->first)) {
        gain -= key->second;
      }
    }
  }

  entry = weight_dict.find(old_mapping2);
  if (entry != weight_dict.end()) {
    for (std::unordered_map<int, int>::const_iterator key = entry->second.begin();
        key != entry->second.end();
        key ++) {
      if (key->first == -1) {
        gain -= key->second;
        continue;
      }
      int first = _GET_0(key->first);
      if (first != node_id1 && mapping[first] == _GET_1(key->first)) {
        gain -= key->second;
      }
    }
  }

  return gain;
}
