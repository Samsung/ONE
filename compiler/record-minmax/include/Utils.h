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

#ifndef __RECORD_MINMAX_UTILS_H__
#define __RECORD_MINMAX_UTILS_H__

#include <luci/IR/CircleNodes.h>

#include <vector>
#include <string>

namespace record_minmax
{

// Return total number of elements of the node's output tensor
uint32_t numElements(const luci::CircleNode *node);

// Return the node's output tensor size in bytes
size_t getTensorSize(const luci::CircleNode *node);

// Read data from file into buffer with specified size in bytes
void readDataFromFile(const std::string &filename, std::vector<char> &data, size_t data_size);

// Throw exception if input has one of the following conditions.
// 1. Have unknown dimension
// 2. Number of elements is 0
void checkInputDimension(const luci::CircleInput *input);

} // namespace record_minmax

#endif // __RECORD_MINMAX_UTILS_H__
