/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_ODC_MINMAX_READER_H__
#define __ONERT_ODC_MINMAX_READER_H__

#include <string>
#include <utility>
#include <vector>

namespace onert
{
namespace odc
{

// File structure
// uint32_t num of runs

// For each run
// uint32_t num of operations
// uint32_t num of inputs

// For each operation
// uint32_t model id
// uint32_t subgraph id
// uint32_t operation id
// float min
// float max

// For each input
// uint32_t model id
// uint32_t subgraph id
// uint32_t input id
// float min
// float max

struct MinMaxVectors
{
  std::vector<float> min_vector;
  std::vector<float> max_vector;
};

class MinMaxReader
{
public:
  MinMaxReader(const std::string &filepath);
  /**
   * @brief Returns minmax recording for op {model_idx, subg_idx, op_idx}
   *
   * @return MinMaxVectors
   */
  MinMaxVectors readOP(uint32_t model_idx, uint32_t subg_idx, uint32_t op_idx) const;
  /**
   * @brief Returns minmax recording for input {model_idx, subg_idx, input_idx}
   *
   * @return MinMaxVectors
   */
  MinMaxVectors readInput(uint32_t model_idx, uint32_t subg_idx, uint32_t input_idx) const;

private:
  std::string _filepath;
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_MINMAX_READER_H__
