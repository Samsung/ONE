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

#include "record-hessian/RecordHessian.h"
#include "record-hessian/HessianObserver.h"

#include <dio_hdf5/HDF5Importer.h>

#include <iostream>

using Shape = std::vector<loco::Dimension>;

namespace
{

uint32_t numElements(const luci::CircleNode *node)
{
  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
    num_elements *= node->dim(i).value();

  return num_elements;
}

[[maybe_unused]] void checkInputDimension(const luci::CircleInput *input)
{
  for (uint32_t i = 0; i < input->rank(); i++)
    if (!input->dim(i).known())
      throw std::runtime_error("RecordHessian: " + input->name() + " has unknown dimension");

  if (numElements(input) == 0)
    throw std::runtime_error("RecordHessian: " + input->name() + " is a zero-sized input");
}

/**
 * @brief  getTensorSize will return size in bytes
 */
template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/**
 * @brief  verifyTypeShape checks the type and the shape of CircleInput
 *         This throws an exception if type or shape does not match
 */
[[maybe_unused]] void verifyTypeShape(const luci::CircleInput *input_node, const loco::DataType &dtype,
                     const Shape &shape)
{
  if (dtype != input_node->dtype())
    throw std::runtime_error("RecordHessian: Wrong input type.");

  if (shape.size() != input_node->rank())
    throw std::runtime_error("RecordHessian: Input rank mismatch.");

  for (uint32_t i = 0; i < shape.size(); i++)
  {
    if (not(shape.at(i) == input_node->dim(i)))
      throw std::runtime_error("RecordHessian: Input shape mismatch.");
  }
}

} // namespace

// namespace record_hessian
// {
// // void RecordHessian::initialize(luci::Module *module); // To Be Implemented
// // std::unique_ptr<HessianMap> RecordHessian::profileData(const std::string &input_data_path); // To Be Implemented
// } // namespace record_hessian
