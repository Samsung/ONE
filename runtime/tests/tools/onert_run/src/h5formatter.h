/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_RUN_H5FORMATTER_H__
#define __ONERT_RUN_H5FORMATTER_H__

#include "allocation.h"
#include "types.h"

#include <string>
#include <vector>

struct nnfw_session;

namespace onert_run
{
class H5Formatter
{
public:
  H5Formatter() = default;
  std::vector<TensorShape> readTensorShapes(const std::string &filename, uint32_t num_inputs);
  void loadInputs(const std::string &filename, std::vector<Allocation> &inputs);
  void dumpOutputs(const std::string &filename, const std::vector<Allocation> &outputs,
                   const std::vector<TensorShape> &shape_map);
};
} // namespace onert_run

#endif // __ONERT_RUN_H5FORMATTER_H__
