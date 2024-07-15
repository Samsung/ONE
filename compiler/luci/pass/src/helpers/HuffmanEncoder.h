/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_PASS_HELPERS_HUFFMAN_ENCODER_H__
#define __LUCI_PASS_HELPERS_HUFFMAN_ENCODER_H__

#include <bitset>
#include <climits>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace luci
{

namespace huffman
{

// Node of prefix tree
template <typename T> struct Node
{
  std::shared_ptr<Node<T>> p_left;
  std::shared_ptr<Node<T>> p_right;
  T data;
  unsigned int freq;
};

// Compare functor for priority queue
template <typename T> struct CompareNodes
{
  bool operator()(std::shared_ptr<Node<T>> l, std::shared_ptr<Node<T>> r)
  {
    return l->freq > r->freq;
  }
};

template <typename T> class HuffmanEncoder
{
private:
  std::unordered_map<T, std::string> _huffman_table;

private:
  std::shared_ptr<Node<T>> allocateNode(T data, unsigned int freq, std::shared_ptr<Node<T>> p_left,
                                        std::shared_ptr<Node<T>> p_right)
  {
    std::shared_ptr<Node<T>> node = std::make_unique<Node<T>>();
    node->data = data;
    node->freq = freq;
    node->p_left = p_left;
    node->p_right = p_right;
    return node;
  }

  std::unordered_map<T, unsigned int> calculateFrequencyMap(const std::vector<T> &input)
  {
    std::unordered_map<T, unsigned int> out_map;
    for (auto &item : input)
      out_map[item] = out_map.find(item) != out_map.end() ? out_map[item] + 1 : 1;

    return out_map;
  }

  std::string exportHuffmanTreeToString(std::shared_ptr<Node<T>> node)
  {
    if (node == nullptr)
      return "";

    if (!node->p_left && !node->p_right)
    {
      return "0" + std::bitset<sizeof(T) * CHAR_BIT>(node->data).to_string();
    }

    std::string tmp = "1";
    tmp += exportHuffmanTreeToString(node->p_left);
    tmp += exportHuffmanTreeToString(node->p_right);
    return tmp;
  }

  void buildHuffmanTable(std::shared_ptr<Node<T>> node, const std::string str = "")
  {
    if (node == nullptr)
      return;

    if (!node->p_left && !node->p_right)
    {
      _huffman_table[node->data] = str;
    }

    buildHuffmanTable(node->p_left, str + "0");
    buildHuffmanTable(node->p_right, str + "1");
  }

  std::shared_ptr<Node<T>> buildHuffmanTree(const std::vector<T> &input)
  {
    auto freq_map = calculateFrequencyMap(input);

    std::priority_queue<std::shared_ptr<Node<T>>, std::vector<std::shared_ptr<Node<T>>>,
                        CompareNodes<T>>
      pq;

    for (auto &item : freq_map)
    {
      pq.push(allocateNode(item.first, item.second, nullptr, nullptr));
    }

    while (pq.size() != 1)
    {
      std::shared_ptr<Node<T>> left = pq.top();
      pq.pop();
      std::shared_ptr<Node<T>> right = pq.top();
      pq.pop();

      unsigned int sum = left->freq + right->freq;
      pq.push(allocateNode(0, sum, left, right));
    }

    return pq.top();
  }

  struct EncodedTreeAndData
  {
    std::vector<bool> tree_vec{};
    std::vector<bool> data_vec{};
  };

  std::vector<uint8_t> packEncodedDataToArray(const std::string &tree_str,
                                              const std::string &encoded_data)
  {
    std::vector<uint8_t> arr;
    const size_t kTreeSizeInBits = tree_str.size();
    const size_t kDataSizeInBits = encoded_data.size();

    for (size_t i = 0; i < sizeof(size_t); ++i)
    {
      arr.push_back(
        *(static_cast<const uint8_t *>(static_cast<const void *>(&kTreeSizeInBits)) + i));
    }

    for (size_t i = 0; i < sizeof(size_t); ++i)
    {
      arr.push_back(
        *(static_cast<const uint8_t *>(static_cast<const void *>(&kDataSizeInBits)) + i));
    }

    const auto merged_str = tree_str + encoded_data;
    const size_t kMergedSizeInBits = merged_str.size();

    const auto kMergedSizeInBytes = kMergedSizeInBits % CHAR_BIT ? kMergedSizeInBits / CHAR_BIT + 1
                                                                 : kMergedSizeInBits / CHAR_BIT;
    for (size_t i = 0; i < kMergedSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kMergedSizeInBits - i * CHAR_BIT < CHAR_BIT ? kMergedSizeInBits - i * CHAR_BIT : CHAR_BIT;

      std::string tmp_str = merged_str.substr(i * CHAR_BIT, kNumOfBits);

      for (size_t i = 0; i < CHAR_BIT - kNumOfBits; ++i)
        tmp_str += "0";

      const std::bitset<CHAR_BIT> tmp_bitset(tmp_str);

      arr.push_back(static_cast<uint8_t>(tmp_bitset.to_ullong()));
    }
    return arr;
  }

public:
  // Encodes input vector of values of type T and returns encoded vector of uint8_t
  std::vector<uint8_t> encode(const std::vector<T> &input)
  {
    std::shared_ptr<Node<T>> root = buildHuffmanTree(input);
    buildHuffmanTable(root);

    std::string exported_tree = exportHuffmanTreeToString(root);
    std::string str = "";

    for (auto &item : input)
    {
      str += _huffman_table[item];
    }

    std::vector<uint8_t> raw_arr = packEncodedDataToArray(exported_tree, str);
    return raw_arr;
  }

public:
  HuffmanEncoder() = default;
};
} // namespace huffman
} // namespace luci
#endif // __LUCI_PASS_HELPERS_HUFFMAN_ENCODER_H__
