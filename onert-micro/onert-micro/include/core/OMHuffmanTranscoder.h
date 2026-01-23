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
  void decode(Node<T> *node, int &index, std::string str)
  {
    if (node == nullptr)
    {
      return;
    }

    if (!node->p_left && !node->p_right)
    {
      std::cout << static_cast<int>(node->data);
      return;
    }

    index++;

    if (str[index] == '0')
      decode(node->p_left, index, str);
    else
      decode(node->p_right, index, str);
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
    std::string tree_str{""};
    std::string data_str{""};
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

    const auto kTreeSizeInBytes =
      kTreeSizeInBits % CHAR_BIT ? kTreeSizeInBits / CHAR_BIT + 1 : kTreeSizeInBits / CHAR_BIT;

    for (int i = 0; i < kTreeSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kTreeSizeInBits - i * CHAR_BIT < CHAR_BIT ? kTreeSizeInBits - i * CHAR_BIT : CHAR_BIT;
      for (int j = 0; j < kNumOfBits; ++j)
      {
        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.tree_str += "1";
        else
          tree_and_data.tree_str += "0";
      }
    }
    const auto kDataSizeInBytes =
      kDataSizeInBits % CHAR_BIT ? kDataSizeInBits / CHAR_BIT + 1 : kDataSizeInBits / CHAR_BIT;
    const auto kOffsetInBits = kTreeSizeInBits % CHAR_BIT;
    start_pos += kOffsetInBits ? kTreeSizeInBytes - 1 : kTreeSizeInBytes;

    for (int i = 0; i < kDataSizeInBytes; ++i)
    {
      const auto kNumOfBits = kDataSizeInBits - i * CHAR_BIT < CHAR_BIT
                                ? kOffsetInBits + kDataSizeInBits - i * CHAR_BIT
                                : CHAR_BIT;
      for (int j = kOffsetInBits; j < kNumOfBits; ++j)
      {

        if (*(pack_ptr + start_pos + i) & ((1 << 7) >> j))
          tree_and_data.data_str += "1";
        else
          tree_and_data.data_str += "0";
      }
      if (kNumOfBits < CHAR_BIT)
        break;
      for (int j = 0; j < kOffsetInBits; ++j)
      {

        if (*(pack_ptr + start_pos + i + 1) & ((1 << 7) >> j))
          tree_and_data.data_str += "1";
        else
          tree_and_data.data_str += "0";
      }
    }
    return tree_and_data;
  }

public:
  void encode_decode_example()
  {
    // std::vector<T> input{

    //};
    std::vector<T> input;
    for (int i = 0; i < 10; ++i)
      input.push_back(0);
    for (int i = 0; i < 2; ++i)
      input.push_back(i);

    // input.insert(input.end(), input.begin(), input.end());

    buildHuffmanTree(input);
    buildHuffmanTable(root);
    auto exported_tree = exportHuffmanTreeToString(root);
    // auto root_imported = importHuffmanTreeFromString(exported_tree);
    std::cout << "Huffman Codes are :\n";
    for (auto pair : huffmanCode)
    {
      std::cout << static_cast<int>(pair.first) << " " << pair.second << '\n';
    }

    std::cout << "\nInput string bits:\n";
    auto input_bits = input.size() * CHAR_BIT * sizeof(T);
    std::cout << input_bits << "\n";

    // TODO: replace string with bitset or bool vector
    // print encoded string
    std::string str = "";
    for (auto &item : input)
    {
      str += huffmanCode[item];
    }
    std::vector<uint8_t> raw_arr = packEncodedDataToArray(exported_tree, str);
    EncodedTreeAndData encoded_tree_and_data = unpackArrayToEncodedTreeAndData(raw_arr);
    std::vector<uint8_t> raw_arr_cmp =
      packEncodedDataToArray(encoded_tree_and_data.tree_str, encoded_tree_and_data.data_str);
    EncodedTreeAndData encoded_tree_and_data_cmp = unpackArrayToEncodedTreeAndData(raw_arr_cmp);
    if ((encoded_tree_and_data.data_str == encoded_tree_and_data_cmp.data_str &&
         encoded_tree_and_data.tree_str == encoded_tree_and_data_cmp.tree_str))
      std::cout << "structs are equal\n";
    else
      std::cout << "structs are different\n";
    if (raw_arr == raw_arr_cmp)
      std::cout << "raws are equal\n";
    else
      std::cout << "raws are different\n";

    // std::cout << "\nEncoded string is :\n" << str << '\n';
    std::cout << "\nEncoded string bits:\n" << str.size() << '\n';
    auto tree_bits = nodes_count + CHAR_BIT * sizeof(T) * huffmanCode.size();
    std::cout << "Bits to store tree:\n" << tree_bits << '\n';
    auto encoded_bits_total = tree_bits + str.size();
    std::cout << "Compression:\n"
              << ((input_bits - encoded_bits_total) / (float)input_bits) * 100 << "% \n";

    // decode the encoded string
    // int index = -1;
    // std::cout << "\nDecoded string is: \n";
    // while (index < (int)str.size() - 2) {
    //	decode(root, index, str);
    //}
  }
  HuffmanTranscoder() = default;
};
} // namespace core
} // namespace onert_micro
#endif // ONERT_MICRO_CORE_OM_HUFFMAN_TRANSCODER_H
