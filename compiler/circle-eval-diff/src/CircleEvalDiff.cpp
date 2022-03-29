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

#include "CircleEvalDiff.h"
#include "ModuleEvalDiff.h"
#include "MetricPrinter.h"

#include <luci/Importer.h>

#include <fstream>
#include <stdexcept>
#include <iostream>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace
{

/**
 * @brief  getTensorSize will return size in bytes
 */
template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/**
 * @brief  verifyTypeShape checks the type and the shape of CircleInput
 *         This throws an exception if type or shape does not match
 */
void verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
{
  // Type check
  if (dtype != input_node->dtype())
    throw std::runtime_error("Wrong input type.");

  if (shape.num_dims() != input_node->rank())
    throw std::runtime_error("Input rank mismatch.");

  for (uint32_t i = 0; i < shape.num_dims(); i++)
  {
    if (shape.dim(i) != input_node->dim(i).value())
      throw std::runtime_error("Input shape mismatch.");
  }
}

std::unique_ptr<luci::Module> import(const std::string &model_path)
{
  // Load model from the file
  std::ifstream fs(model_path, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + model_path + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify circle '" + model_path + "'");
  }

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (module == nullptr)
    throw std::runtime_error("Failed to load '" + model_path + "'");

  return module;
}

uint32_t getByteSize(const luci::CircleNode *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

} // namespace

namespace circle_eval_diff
{

CircleEvalDiff::CircleEvalDiff(std::unique_ptr<Context> &&ctx)
  : _ctx(std::move(ctx)), _runner(nullptr)
{
}

CircleEvalDiff::~CircleEvalDiff() = default;

void CircleEvalDiff::init()
{
  // Set metric
  std::unique_ptr<MetricPrinter> metric;
  switch (_ctx->metric)
  {
    case Metric::MAE:
      metric = std::make_unique<MAEPrinter>();
      break;
    default:
      throw std::runtime_error("Unsupported metric.");
  }

  auto first_module = import(_ctx->first_model_path);
  auto second_module = import(_ctx->second_model_path);

  // Set runner
  switch (_ctx->input_format)
  {
    case InputFormat::H5:
      _runner = std::make_unique<H5InputEvalDiff>(std::move(first_module), std::move(second_module),
                                                  std::move(metric));
      break;
    default:
      throw std::runtime_error("Unsupported input format.");
  }
}

void CircleEvalDiff::evalDiff(const std::string &first_input_data_path,
                              const std::string &second_input_data_path) const
{
  _runner->evalDiff(first_input_data_path, second_input_data_path);
}

} // namespace circle_eval_diff
