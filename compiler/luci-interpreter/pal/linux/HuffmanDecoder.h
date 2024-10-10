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

#ifndef LUCI_INTERPRETER_PAL_HUFFMAN_DECODER_H
#define LUCI_INTERPRETER_PAL_HUFFMAN_DECODER_H

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

namespace luci_interpreter_pal
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
  std::shared_ptr<Node<T>> importHuffmanTreeFromBoolVec(std::vector<bool> &vec, size_t &index)
  {
    if (vec.empty())
      return nullptr;
    if (vec[index])
    {
      index++;
      std::shared_ptr<Node<T>> p_left = importHuffmanTreeFromBoolVec(vec, index);
      std::shared_ptr<Node<T>> p_right = importHuffmanTreeFromBoolVec(vec, index);
      return createNode<T>(0, 0, p_left, p_right);
    }
    else if (vec[index] == false)
    {
      index++;
      T tmp = 0;
      for (size_t i = 0; i < sizeof(T) * CHAR_BIT; ++i)
      {
        if (vec[index++])
          tmp |= (1 << (sizeof(T) * CHAR_BIT - 1)) >> i;
      }

      return createNode<T>(tmp, 0, nullptr, nullptr);
    }
    return nullptr;
  }

  EncodedTreeAndData unpackArrayToEncodedTreeAndData(const uint8_t *pack_ptr)
  {
    constexpr auto kTreeSizeBytesN = sizeof(size_t);
    constexpr auto kDataSizeBytesN = sizeof(size_t);

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

    for (size_t i = 0; i < kTreeSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kTreeSizeInBits - i * CHAR_BIT < CHAR_BIT ? kTreeSizeInBits - i * CHAR_BIT : CHAR_BIT;
      for (size_t j = 0; j < kNumOfBits; ++j)
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

    for (size_t i = 0; i < kDataSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kDataSizeInBits - i * CHAR_BIT < CHAR_BIT ? kDataSizeInBits - i * CHAR_BIT : CHAR_BIT;
      const auto kBitsInFirstByteToRead =
        kNumOfBits < CHAR_BIT - kOffsetInBits ? kNumOfBits : CHAR_BIT - kOffsetInBits;
      for (size_t j = kOffsetInBits; j < kOffsetInBits + kBitsInFirstByteToRead; ++j)
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
      for (size_t j = 0; j < kBitsLeft; ++j)
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

    const auto kTreeSizeInBytes =
      kTreeSizeInBits % CHAR_BIT ? kTreeSizeInBits / CHAR_BIT + 1 : kTreeSizeInBits / CHAR_BIT;

    for (size_t i = 0; i < kTreeSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kTreeSizeInBits - i * CHAR_BIT < CHAR_BIT ? kTreeSizeInBits - i * CHAR_BIT : CHAR_BIT;
      for (size_t j = 0; j < kNumOfBits; ++j)
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

    for (size_t i = 0; i < kDataSizeInBytes; ++i)
    {
      const auto kNumOfBits =
        kDataSizeInBits - i * CHAR_BIT < CHAR_BIT ? kDataSizeInBits - i * CHAR_BIT : CHAR_BIT;
      const auto kBitsInFirstByteToRead =
        kNumOfBits < CHAR_BIT - kOffsetInBits ? kNumOfBits : CHAR_BIT - kOffsetInBits;
      for (size_t j = kOffsetInBits; j < kOffsetInBits + kBitsInFirstByteToRead; ++j)
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
      for (size_t j = 0; j < kBitsLeft; ++j)
      {
        if (*(pack_ptr + start_pos + i + 1) & ((1 << 7) >> j))
          tree_and_data.data_vec.push_back(true);
        else
          tree_and_data.data_vec.push_back(false);
      }
    }
    return tree_and_data;
  }

  void decode(std::shared_ptr<Node<T>> node, std::vector<bool> &vec, uint8_t *dst_ptr)
  {
    if (node == nullptr)
    {
      return;
    }

    if (!node->p_left && !node->p_right)
    {
      *dst_ptr = node->data;
      return;
    }

    if (vec.size() == _decode_idx)
      return;
    if (vec[_decode_idx] == false)
    {
      ++_decode_idx;
      decode(node->p_left, vec, dst_ptr);
    }
    else
    {
      ++_decode_idx;
      decode(node->p_right, vec, dst_ptr);
    }
  }

public:
  HuffmanDecoder() = default;

  void init_decoder(const uint8_t *input)
  {
    size_t index = 0;
    _encoded_tree_and_data = unpackArrayToEncodedTreeAndData(input);
    _root = importHuffmanTreeFromBoolVec(_encoded_tree_and_data.tree_vec, index);
  }

  void reset_decode_idx(void) { _decode_idx = 0; }

  size_t decode_n(uint8_t *dst_ptr, size_t num)
  {
    size_t bytes_decoded = 0;
    for (size_t i = 0; i < num && _decode_idx < _encoded_tree_and_data.data_vec.size(); ++i)
    {
      decode(_root, _encoded_tree_and_data.data_vec, dst_ptr + bytes_decoded);
      bytes_decoded++;
    }
    return bytes_decoded;
  }

  std::vector<T> decode(const std::vector<uint8_t> &input)
  {
    init_decoder(input.data());
    std::vector<T> decoded{};
    T tmp;
    while (decode_n(reinterpret_cast<uint8_t *>(&tmp), sizeof(T)))
    {
      decoded.push_back(tmp);
    }
    return decoded;
  }
};

} // namespace huffman
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_HUFFMAN_DECODER_H
