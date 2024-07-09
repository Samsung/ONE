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

#ifndef ONERT_MICRO_CORE_OM_HUFFMAN_TRANSCODER_H
#define ONERT_MICRO_CORE_OM_HUFFMAN_TRANSCODER_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <queue>
#include <string>
#include <bitset>
#include <climits>

namespace onert_micro
{
namespace core
{
template <typename T> struct Node
{
  Node *p_left = nullptr;
  Node *p_right = nullptr;
  T data;
  unsigned int freq;
};

template <typename T> struct CompareNodes
{
  bool operator()(Node<T> *l, Node<T> *r) { return l->freq > r->freq; }
};

template <typename T> class HuffmanTranscoder
{
private:
  Node<T> *root = nullptr;
  std::unordered_map<T, std::string> huffmanCode;
  std::vector<bool> encoded_bitset{};
  std::size_t nodes_count = 0;

private:
  Node<T> *allocateNode(T data, unsigned int freq, Node<T> *p_left, Node<T> *p_right)
  {
    Node<T> *node = new Node<T>;
    node->data = data;
    node->freq = freq;
    node->p_left = p_left;
    node->p_right = p_right;
    nodes_count++;
    return node;
  }

  std::unordered_map<T, unsigned int> calculate_frequency_map(const std::vector<T> &input)
  {
    std::unordered_map<T, unsigned int> out_map;
    for (auto &item : input)
      out_map[item] = out_map.find(item) != out_map.end() ? out_map[item] + 1 : 1;
    return out_map;
  }

  std::string exportHuffmanTreeToString(Node<T> *node)
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

  Node<T> *importHuffmanTreeFromBoolVec(std::vector<bool> &vec, size_t &index)
  {
    if (vec.empty())
      return nullptr;
    if (vec[index])
    {
      index++;
      Node<T> *p_left = importHuffmanTreeFromBoolVec(vec, index);
      Node<T> *p_right = importHuffmanTreeFromBoolVec(vec, index);
      return allocateNode(0, 0, p_left, p_right);
    }
    else if (vec[index] == false)
    {
      index++;
      T tmp = 0;
      for (int i = 0; i < sizeof(T) * CHAR_BIT; ++i)
      {
        if (vec[index++])
          tmp |= (1 << (sizeof(T) * CHAR_BIT - 1)) >> i;
      }

      return allocateNode(tmp, 0, nullptr, nullptr);
    }
  }

  Node<T> *importHuffmanTreeFromString(std::string &str)
  {

    if (str.substr(0, 1) == "1")
    {
      str = str.substr(1);
      Node<T> *p_left = importHuffmanTreeFromString(str);
      Node<T> *p_right = importHuffmanTreeFromString(str);
      return allocateNode(0, 0, p_left, p_right);
    }
    else if (str.substr(0, 1) == "0")
    {
      str = str.substr(1);
      std::bitset<sizeof(T) * CHAR_BIT> tmp(str.substr(0, sizeof(T) * CHAR_BIT));
      str = str.substr(sizeof(T) * CHAR_BIT);
      return allocateNode(static_cast<T>(tmp.to_ullong()), 0, nullptr, nullptr);
    }
  }

  void buildHuffmanTable(Node<T> *node, const std::string str = "")
  {
    if (node == nullptr)
      return;

    if (!node->p_left && !node->p_right)
    {
      huffmanCode[node->data] = str;
    }

    buildHuffmanTable(node->p_left, str + "0");
    buildHuffmanTable(node->p_right, str + "1");
  }

  void decode(Node<T> *node, std::string &str, std::vector<T> &out_vec, size_t &index)
  {
    if (node == nullptr)
    {
      return;
    }

    if (!node->p_left && !node->p_right)
    {
      out_vec.push_back(node->data);
      return;
    }

    if (str.size() == index)
      return;
    if (str[index] == '0')
    {
      // str = str.substr(0, str.size() - 1);
      decode(node->p_left, str, out_vec, ++index);
    }
    else
    {
      // str = str.substr(0, str.size() - 1);
      decode(node->p_right, str, out_vec, ++index);
    }
  }

  void buildHuffmanTree(const std::vector<T> &input)
  {
    auto freq_map = calculate_frequency_map(input);

    std::priority_queue<Node<T> *, std::vector<Node<T> *>, CompareNodes<T>> pq;

    for (auto &item : freq_map)
    {
      pq.push(allocateNode(item.first, item.second, nullptr, nullptr));
    }

    while (pq.size() != 1)
    {
      Node<T> *left = pq.top();
      pq.pop();
      Node<T> *right = pq.top();
      pq.pop();

      unsigned int sum = left->freq + right->freq;
      pq.push(allocateNode(0, sum, left, right));
    }

    root = pq.top();
  }

  struct EncodedTreeAndData
  {
    std::vector<bool> tree_vec{};
    std::vector<bool> data_vec{};
  };

  std::vector<uint8_t> packEncodedDataToArray(const std::string &tree_str,
                                              const std::string &encoded_data)
  {
    constexpr auto kTreeSizeBytesN = sizeof(size_t);
    constexpr auto kDataSizeBytesN = sizeof(size_t);
    std::vector<uint8_t> arr;
    const size_t kTreeSizeInBits = tree_str.size();
    const size_t kDataSizeInBits = encoded_data.size();
    for (int i = 0; i < sizeof(size_t); ++i)
    {
      arr.push_back(
        *(static_cast<const uint8_t *>(static_cast<const void *>(&kTreeSizeInBits)) + i));
    }
    for (int i = 0; i < sizeof(size_t); ++i)
    {
      arr.push_back(
        *(static_cast<const uint8_t *>(static_cast<const void *>(&kDataSizeInBits)) + i));
    }
    const auto merged_str = tree_str + encoded_data;
    const size_t kMergedSizeInBits = merged_str.size();

    const auto kMergedSizeInBytes = kMergedSizeInBits % CHAR_BIT ? kMergedSizeInBits / CHAR_BIT + 1
                                                                 : kMergedSizeInBits / CHAR_BIT;
    for (int i = 0; i < kMergedSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kMergedSizeInBits - i * CHAR_BIT < CHAR_BIT ? kMergedSizeInBits - i * CHAR_BIT : CHAR_BIT;
      std::string tmp_str = merged_str.substr(i * CHAR_BIT, kNumOfBits);
      for (int i = 0; i < CHAR_BIT - kNumOfBits; ++i)
        tmp_str += "0";
      const std::bitset<CHAR_BIT> tmp_bitset(tmp_str);
      arr.push_back(static_cast<uint8_t>(tmp_bitset.to_ullong()));
    }
    return arr;
  }

  EncodedTreeAndData unpackArrayToEncodedTreeAndData(const uint8_t *pack_ptr)
  {
    constexpr auto kTreeSizeBytesN = sizeof(size_t);
    constexpr auto kDataSizeBytesN = sizeof(size_t);
    //    const uint8_t *pack_ptr = packed_vec.data();
    const std::bitset<CHAR_BIT * kTreeSizeBytesN> tree_size_bitset(
      *static_cast<const size_t *>(static_cast<const void *>(pack_ptr)));
    const std::bitset<CHAR_BIT * kDataSizeBytesN> data_size_bitset(
      *static_cast<const size_t *>(static_cast<const void *>(pack_ptr + kTreeSizeBytesN)));

    const size_t kTreeSizeInBits = static_cast<size_t>(tree_size_bitset.to_ullong());
    const size_t kDataSizeInBits = static_cast<size_t>(data_size_bitset.to_ullong());

    auto start_pos = kTreeSizeBytesN + kDataSizeBytesN;
    EncodedTreeAndData tree_and_data;

    const auto kTreeSizeInBytes =
      kTreeSizeInBits % CHAR_BIT ? kTreeSizeInBits / CHAR_BIT + 1 : kTreeSizeInBits / CHAR_BIT;

    for (int i = 0; i < kTreeSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kTreeSizeInBits - i * CHAR_BIT < CHAR_BIT ? kTreeSizeInBits - i * CHAR_BIT : CHAR_BIT;
      for (int j = 0; j < kNumOfBits; ++j)
      {
        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.tree_vec.push_back(true);
        else
          tree_and_data.tree_vec.push_back(false);
      }
    }
    const auto kDataSizeInBytes =
      kDataSizeInBits % CHAR_BIT ? kDataSizeInBits / CHAR_BIT + 1 : kDataSizeInBits / CHAR_BIT;
    const auto kOffsetInBits = kTreeSizeInBits % CHAR_BIT;
    start_pos += kOffsetInBits ? kTreeSizeInBytes - 1 : kTreeSizeInBytes;

    for (int i = 0; i < kDataSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kDataSizeInBits - i * CHAR_BIT < CHAR_BIT ? kDataSizeInBits - i * CHAR_BIT : CHAR_BIT;
      const auto kBitsInFirstByteToRead =
        kNumOfBits < CHAR_BIT - kOffsetInBits ? kNumOfBits : CHAR_BIT - kOffsetInBits;
      for (int j = kOffsetInBits; j < kOffsetInBits + kBitsInFirstByteToRead; ++j)
      {

        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
      if (kNumOfBits < CHAR_BIT - kOffsetInBits)
        break;
      const auto kBitsLeft = kNumOfBits - (CHAR_BIT - kOffsetInBits) < kOffsetInBits
                               ? kNumOfBits - (CHAR_BIT - kOffsetInBits)
                               : kOffsetInBits;
      for (int j = 0; j < kBitsLeft; ++j)
      {

        if (*(pack_ptr + start_pos + i + 1) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
    }
    return tree_and_data;
  }

  EncodedTreeAndData unpackArrayToEncodedTreeAndData(const std::vector<uint8_t> &packed_vec)
  {
    constexpr auto kTreeSizeBytesN = sizeof(size_t);
    constexpr auto kDataSizeBytesN = sizeof(size_t);
    const uint8_t *pack_ptr = packed_vec.data();
    const std::bitset<CHAR_BIT * kTreeSizeBytesN> tree_size_bitset(
      *static_cast<const size_t *>(static_cast<const void *>(pack_ptr)));
    const std::bitset<CHAR_BIT * kDataSizeBytesN> data_size_bitset(
      *static_cast<const size_t *>(static_cast<const void *>(pack_ptr + kTreeSizeBytesN)));

    const size_t kTreeSizeInBits = static_cast<size_t>(tree_size_bitset.to_ullong());
    const size_t kDataSizeInBits = static_cast<size_t>(data_size_bitset.to_ullong());

    auto start_pos = kTreeSizeBytesN + kDataSizeBytesN;
    EncodedTreeAndData tree_and_data;
    //                tree_and_data.tree_vec.push_back.reserve(kTreeSizeInBits);
    //                tree_and_data.data_vec.push_back.reserve(kDataSizeInBits);

    const auto kTreeSizeInBytes =
      kTreeSizeInBits % CHAR_BIT ? kTreeSizeInBits / CHAR_BIT + 1 : kTreeSizeInBits / CHAR_BIT;

    for (int i = 0; i < kTreeSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kTreeSizeInBits - i * CHAR_BIT < CHAR_BIT ? kTreeSizeInBits - i * CHAR_BIT : CHAR_BIT;
      for (int j = 0; j < kNumOfBits; ++j)
      {
        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
    }
    const auto kDataSizeInBytes =
      kDataSizeInBits % CHAR_BIT ? kDataSizeInBits / CHAR_BIT + 1 : kDataSizeInBits / CHAR_BIT;
    const auto kOffsetInBits = kTreeSizeInBits % CHAR_BIT;
    start_pos += kOffsetInBits ? kTreeSizeInBytes - 1 : kTreeSizeInBytes;

    for (int i = 0; i < kDataSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kDataSizeInBits - i * CHAR_BIT < CHAR_BIT ? kDataSizeInBits - i * CHAR_BIT : CHAR_BIT;
      const auto kBitsInFirstByteToRead =
        kNumOfBits < CHAR_BIT - kOffsetInBits ? kNumOfBits : CHAR_BIT - kOffsetInBits;
      for (int j = kOffsetInBits; j < kOffsetInBits + kBitsInFirstByteToRead; ++j)
      {

        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
      if (kNumOfBits < CHAR_BIT - kOffsetInBits)
        break;
      const auto kBitsLeft = kNumOfBits - (CHAR_BIT - kOffsetInBits) < kOffsetInBits
                               ? kNumOfBits - (CHAR_BIT - kOffsetInBits)
                               : kOffsetInBits;
      for (int j = 0; j < kBitsLeft; ++j)
      {

        if (*(pack_ptr + start_pos + i + 1) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
    }
    return tree_and_data;
  }

public:
  std::vector<uint8_t> encodeInputArray(const std::vector<T> &input)
  {
    buildHuffmanTree(input);
    buildHuffmanTable(root);
    auto exported_tree = exportHuffmanTreeToString(root);
    std::string str = "";
    for (auto &item : input)
    {
      str += huffmanCode[item];
    }
    std::vector<uint8_t> raw_arr = packEncodedDataToArray(exported_tree, str);
    return raw_arr;
  }

  void decode(Node<T> *node, std::vector<bool> &vec, T *dst_ptr)
  {
    if (node == nullptr)
    {
      return;
    }

    if (!node->p_left && !node->p_right)
    {
      *dst_ptr = node->data;
      // dst_ptr++;
      return;
    }

    if (vec.size() == _decode_idx)
      return;
    if (vec[_decode_idx] == false)
    {
      // str = str.substr(0, str.size() - 1);
      ++_decode_idx;
      decode(node->p_left, vec, dst_ptr);
    }
    else
    {
      // str = str.substr(0, str.size() - 1);
      ++_decode_idx;
      decode(node->p_right, vec, dst_ptr);
    }
  }

private:
  size_t _decode_idx = 0;
  EncodedTreeAndData _encoded_tree_and_data;

public:
  void init_decoder(const uint8_t *input)
  {
    size_t index = 0;
    _encoded_tree_and_data = unpackArrayToEncodedTreeAndData(input);
    root = importHuffmanTreeFromBoolVec(_encoded_tree_and_data.tree_vec, index);
  }

  void reset_decode_idx(void) { _decode_idx = 0; }

  int decode_n(uint8_t *dst_ptr, size_t num)
  {
    // EncodedTreeAndData encoded_tree_and_data = unpackArrayToEncodedTreeAndData(input);
    // auto root_imported = importHuffmanTreeFromString(encoded_tree_and_data.tree_str);
    /*size_t index = 0;*/
    size_t bytes_decoded = 0;
    for (int i = 0; i < num && _decode_idx < _encoded_tree_and_data.data_vec.size(); ++i)
    {
      decode(root, _encoded_tree_and_data.data_vec, dst_ptr + bytes_decoded);
      bytes_decoded++;
    }
    return bytes_decoded;
  }

  std::vector<T> decodeEncodedArray(const uint8_t *input)
  {

    size_t index = 0;
    std::vector<uint8_t> res{};
    // std::reverse(encoded_tree_and_data.data_str.begin(), encoded_tree_and_data.data_str.end());

    while (index < _encoded_tree_and_data.data_str.size())
      decode(root, _encoded_tree_and_data.data_str, res, index);
    return res;
  }

  std::vector<T> decodeEncodedArray(const std::vector<uint8_t> &input)
  {
    EncodedTreeAndData encoded_tree_and_data = unpackArrayToEncodedTreeAndData(input);
    auto root_imported = importHuffmanTreeFromString(encoded_tree_and_data.tree_str);
    size_t index = 0;
    std::vector<uint8_t> res{};
    // std::reverse(encoded_tree_and_data.data_str.begin(), encoded_tree_and_data.data_str.end());

    while (index < encoded_tree_and_data.data_str.size())
      decode(root_imported, encoded_tree_and_data.data_str, res, index);
    return res;
  }

  HuffmanTranscoder() = default;
};
} // namespace core
} // namespace onert_micro
#endif // ONERT_MICRO_CORE_OM_HUFFMAN_TRANSCODER_H
