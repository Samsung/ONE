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

#include "tflite_loader.h"
#include "base_loader.h"
#include "tflite_schema_generated.h"

namespace onert
{
namespace tflite_loader
{

namespace
{

struct LoaderDomain
{
  using Verifier = flatbuffers::Verifier;
  using ActivationFunctionType = onert_tflite::ActivationFunctionType;
  using Buffer = onert_tflite::Buffer;
  using BuiltinOperator = onert_tflite::BuiltinOperator;
  using CustomOptionsFormat = onert_tflite::CustomOptionsFormat;
  using Model = onert_tflite::Model;
  using Operator = onert_tflite::Operator;
  using Padding = onert_tflite::Padding;
  using Pool2DOptions = onert_tflite::Pool2DOptions;
  using Tensor = onert_tflite::Tensor;
  using TensorType = onert_tflite::TensorType;
  using SubGraph = onert_tflite::SubGraph;

  static const char *EnumNameBuiltinOperator(BuiltinOperator e)
  {
    return onert_tflite::EnumNameBuiltinOperator(e);
  }
  static const char *EnumNameActivationFunctionType(ActivationFunctionType e)
  {
    return onert_tflite::EnumNameActivationFunctionType(e);
  }
  static const char *EnumNameTensorType(TensorType e)
  {
    return onert_tflite::EnumNameTensorType(e);
  }
  static const Model *GetModel(const void *buf) { return onert_tflite::GetModel(buf); }
  static bool VerifyModelBuffer(Verifier &verifier)
  {
    return onert_tflite::VerifyModelBuffer(verifier);
  }
};

class TFLiteLoader final : public base_loader::BaseLoader<LoaderDomain, TFLiteLoader>
{
public:
  using BaseLoader::BaseLoader;

  void loadSubgraph(const onert_tflite::SubGraph *subgraph)
  {
    // Load tensors
    _tensor_to_operand.resize(subgraph->tensors()->size());
    for (flatbuffers::uoffset_t i = 0; i < subgraph->tensors()->size(); ++i)
    {
      _tensor_to_operand[i] = loadOperand(subgraph->tensors()->Get(i));
    }
    // Set inputs
    for (const std::int32_t input_ind : *subgraph->inputs())
    {
      _graph.addInput(_tensor_to_operand[input_ind]);
    }
    // Set outputs
    for (const std::int32_t output_ind : *subgraph->outputs())
    {
      _graph.addOutput(_tensor_to_operand[output_ind]);
    }
    // Create operations
    for (const auto *op : *subgraph->operators())
    {
      loadOperation(op);
    }
  }
};

} // namespace

std::unique_ptr<ir::Graph> loadModel(const char *filename)
{
  auto graph = std::make_unique<ir::Graph>();
  TFLiteLoader loader(*graph);
  loader.loadFromFile(filename);
  return graph;
}

} // namespace tflite_loader
} // namespace onert
