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

#include "HuffmanEncoder.h"
#include "HuffmanDecoder.h"

#include <vector>

#include <gtest/gtest.h>

namespace
{
std::vector<int8_t> input_s8{13, 17, -8, -8, 1, 84, 33, 53, -26, 26, -14, -1, 23, 59, 28, -8};
std::vector<uint8_t> input_u8{13, 17, 218, 8, 1, 84, 33, 53, 26, 26, 14, 1, 23, 59, 28, 8};

} // namespace

TEST(HuffmanEncodeDecodeTest, simple_test_s8)
{
  luci::huffman::HuffmanEncoder<int8_t> encoder;
  luci::huffman::HuffmanDecoder<int8_t> decoder;

  std::vector<uint8_t> encoded = encoder.encode(input_s8);
  std::vector<int8_t> decoded = decoder.decode(encoded);

  EXPECT_EQ(input_s8, decoded);
}

TEST(HuffmanEncodeDecodeTest, simple_test_u8)
{
  luci::huffman::HuffmanEncoder<uint8_t> encoder;
  luci::huffman::HuffmanDecoder<uint8_t> decoder;

  std::vector<uint8_t> encoded = encoder.encode(input_u8);
  std::vector<uint8_t> decoded = decoder.decode(encoded);

  EXPECT_EQ(input_u8, decoded);
}
