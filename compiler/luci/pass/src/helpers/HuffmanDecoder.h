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

#ifndef __LUCI_PASS_HELPERS_HUFFMAN_DECODER_H__
#define __LUCI_PASS_HELPERS_HUFFMAN_DECODER_H__

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <queue>
#include <string>
#include <bitset>
#include <climits>

#include "HuffmanCommon.h"

namespace luci
{

namespace huffman
{

template <typename T> class HuffmanDecoder
{

private:
  struct EncodedTreeAndData
  {
    std::vector<bool> tree_vec{};
    std::vector<bool> data_vec{};
  };

private:
  std::shared_ptr<Node<T>> _root = nullptr;
  size_t _decode_idx = 0;
  EncodedTreeAndData _encoded_tree_and_data;

private:
  std::shared_ptr<Node<T>> importHuffmanTreeFromBoolVec(std::vector<bool> &vec, size_t &index);
  EncodedTreeAndData unpackArrayToEncodedTreeAndData(const uint8_t *pack_ptr);
  EncodedTreeAndData unpackArrayToEncodedTreeAndData(const std::vector<uint8_t> &packed_vec);
  void decode(std::shared_ptr<Node<T>> node, std::vector<bool> &vec, uint8_t *dst_ptr);

public:
  HuffmanDecoder() = default;

  void init_decoder(const uint8_t *input);
  void reset_decode_idx(void) { _decode_idx = 0; }
  size_t decode_n(uint8_t *dst_ptr, size_t num);

  std::vector<T> decode(const std::vector<uint8_t> &input);
};

} // namespace huffman
} // namespace luci

#endif // __LUCI_PASS_HELPERS_HUFFMAN_DECODER_H__
