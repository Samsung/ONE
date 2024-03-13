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

  void buildHuffmanTable(Node<T> *node, const std::string str)
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

public:
  void encode_decode_example()
  {

    std::vector<T> input;
    for (int i = 0; i < 1024; ++i)
      input.push_back(0);
    for (int i = 0; i < 256; ++i)
      input.push_back(i);

    buildHuffmanTree(input);
    buildHuffmanTable(root, "");

    // std::cout << "Huffman Codes are :\n";
    // for (auto pair : huffmanCode) {
    //	std::cout << static_cast<int>(pair.first) << " " << pair.second << '\n';
    // }

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
