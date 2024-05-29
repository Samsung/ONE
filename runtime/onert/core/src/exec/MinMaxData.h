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

#ifndef __ONERT_EXEC_MINMAX_DATA_H__
#define __ONERT_EXEC_MINMAX_DATA_H__

#include "exec/MinMaxMap.h"

#include <string>

namespace onert
{
namespace exec
{

// Because IOMinMaxMap and OpMinMaxMap does not have the ordering and size information,
// we need to dump model, subgraph id for each minmax

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

class RawMinMaxDumper
{
public:
  RawMinMaxDumper(const std::string &filename);
  /**
   * @brief Dump input minmax map
   *
   * @param[in] in_minmax  input minmax map
   * @param[in] op_minmax  op minmax map
   */

  void dump(const exec::IOMinMaxMap &in_minmax, const exec::OpMinMaxMap &op_minmax) const;

private:
  std::string _filename;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_MINMAX_DATA_H__
