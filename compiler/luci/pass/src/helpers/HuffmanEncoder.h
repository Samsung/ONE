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
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "HuffmanCommon.h"

namespace luci
{
namespace huffman
{

template <typename T> class HuffmanEncoder
{
private:
  std::unordered_map<T, std::string> _huffman_table;

private:
  std::unordered_map<T, unsigned int> calculateFrequencyMap(const std::vector<T> &input);

  std::string exportHuffmanTreeToString(std::shared_ptr<Node<T>> node);

  void buildHuffmanTable(std::shared_ptr<Node<T>> node, const std::string str = "");

  std::shared_ptr<Node<T>> buildHuffmanTree(const std::vector<T> &input);

  std::vector<uint8_t> packEncodedDataToArray(const std::string &tree_str,
                                              const std::string &encoded_data);

public:
  // Encodes input vector of values of type T and returns encoded vector of uint8_t
  std::vector<uint8_t> encode(const std::vector<T> &input);

public:
  HuffmanEncoder() = default;
};

} // namespace huffman
} // namespace luci

#endif // __LUCI_PASS_HELPERS_HUFFMAN_ENCODER_H__
