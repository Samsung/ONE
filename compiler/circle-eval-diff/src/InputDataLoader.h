/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_EVAL_DIFF_INPUT_DATA_LOADER_H__
#define __CIRCLE_EVAL_DIFF_INPUT_DATA_LOADER_H__

#include <dio_hdf5/HDF5Importer.h>
#include <loco/IR/Node.h>
#include <luci/IR/CircleNodes.h>

#include "Tensor.h"

#include <memory>
#include <string>

namespace circle_eval_diff
{

void verifyTypeShape(const luci::CircleInput *input_node, const loco::DataType &dtype,
                     const std::vector<loco::Dimension> &shape);

} // namespace circle_eval_diff

namespace circle_eval_diff
{

enum class InputFormat
{
  Undefined, // For debugging
  H5,
  // TODO Implement Random, Directory
};

class InputDataLoader
{
public:
  using Data = std::vector<Tensor>;

public:
  virtual ~InputDataLoader() = default;

public:
  virtual uint32_t size(void) const = 0;

public:
  virtual Data get(uint32_t data_idx) const = 0;
};

class HDF5Loader final : public InputDataLoader
{
public:
  HDF5Loader(const std::string &file_path, const std::vector<loco::Node *> &input_nodes);

public:
  uint32_t size(void) const final;
  Data get(uint32_t data_idx) const final;

private:
  const std::vector<loco::Node *> _input_nodes;
  std::unique_ptr<dio::hdf5::HDF5Importer> _hdf5;
};

std::unique_ptr<InputDataLoader> makeDataLoader(const std::string &file_path,
                                                const InputFormat &format,
                                                const std::vector<loco::Node *> &input_nodes);

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_INPUT_DATA_LOADER_H__
