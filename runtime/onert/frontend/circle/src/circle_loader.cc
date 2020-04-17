/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "circle_loader.h"
#include "base_loader.h"
#include "circle_schema_generated.h"

namespace onert
{
namespace circle_loader
{

namespace
{

struct LoaderDomain
{
  using Verifier = flatbuffers::Verifier;
  using ActivationFunctionType = circle::ActivationFunctionType;
  using Buffer = circle::Buffer;
  using BuiltinOperator = circle::BuiltinOperator;
  using CustomOptionsFormat = circle::CustomOptionsFormat;
  using Model = circle::Model;
  using Operator = circle::Operator;
  using Padding = circle::Padding;
  using Pool2DOptions = circle::Pool2DOptions;
  using Tensor = circle::Tensor;
  using TensorType = circle::TensorType;
  using SubGraph = circle::SubGraph;

  static const char *EnumNameBuiltinOperator(BuiltinOperator e)
  {
    return circle::EnumNameBuiltinOperator(e);
  }
  static const char *EnumNameActivationFunctionType(ActivationFunctionType e)
  {
    return circle::EnumNameActivationFunctionType(e);
  }
  static const char *EnumNameTensorType(TensorType e) { return circle::EnumNameTensorType(e); }
  static const Model *GetModel(const void *buf) { return circle::GetModel(buf); }
  static bool VerifyModelBuffer(Verifier &verifier) { return circle::VerifyModelBuffer(verifier); }
};

class CircleLoader final : public base_loader::BaseLoader<LoaderDomain, CircleLoader>
{
public:
  using BaseLoader::BaseLoader;

  std::unique_ptr<ir::Graph> loadSubgraph(const circle::SubGraph *circle_subg)
  {
    auto subg = std::make_unique<ir::Graph>();
    // Load tensors
    _tensor_to_operand.resize(circle_subg->tensors()->size());
    for (flatbuffers::uoffset_t i = 0; i < circle_subg->tensors()->size(); ++i)
    {
      _tensor_to_operand[i] = loadOperand(circle_subg->tensors()->Get(i), *subg);
    }
    // Set inputs
    for (const std::int32_t input_ind : *circle_subg->inputs())
    {
      subg->addInput(_tensor_to_operand[input_ind]);
    }
    // Set outputs
    for (const std::int32_t output_ind : *circle_subg->outputs())
    {
      subg->addOutput(_tensor_to_operand[output_ind]);
    }
    // Create operations
    for (const auto *op : *circle_subg->operators())
    {
      CircleLoader::loadOperation(op, *subg);
    }

    (void)circle_subg->data_format();

    subg->finishBuilding();

    return subg;
  }

  void loadOperation(const circle::Operator *op, ir::Graph &subg)
  {
    const auto builtin_op = _model->operator_codes()->Get(op->opcode_index())->builtin_code();

    switch (builtin_op)
    {
      case circle::BuiltinOperator::BuiltinOperator_INSTANCE_NORM:
        loadInstanceNorm(op, subg);
        return;
      default:
        BaseLoader::loadOperation(op, subg);
        return;
    }
  }
};

} // namespace

std::unique_ptr<ir::Graph> loadModel(const char *filename)
{
  auto primary_subgraph = std::make_unique<ir::Graph>();
  CircleLoader loader(primary_subgraph);
  loader.loadFromFile(filename);
  return primary_subgraph;
}

} // namespace circle_loader
} // namespace onert
