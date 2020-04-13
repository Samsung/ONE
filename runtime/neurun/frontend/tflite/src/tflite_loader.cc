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

namespace neurun
{
namespace tflite_loader
{

namespace
{

struct LoaderDomain
{
  using Verifier = flatbuffers::Verifier;
  using ActivationFunctionType = neurun_tflite::ActivationFunctionType;
  using Buffer = neurun_tflite::Buffer;
  using BuiltinOperator = neurun_tflite::BuiltinOperator;
  using CustomOptionsFormat = neurun_tflite::CustomOptionsFormat;
  using Model = neurun_tflite::Model;
  using Operator = neurun_tflite::Operator;
  using Padding = neurun_tflite::Padding;
  using Pool2DOptions = neurun_tflite::Pool2DOptions;
  using Tensor = neurun_tflite::Tensor;
  using TensorType = neurun_tflite::TensorType;
  using SubGraph = neurun_tflite::SubGraph;

  static const char *EnumNameBuiltinOperator(BuiltinOperator e)
  {
    return neurun_tflite::EnumNameBuiltinOperator(e);
  }
  static const char *EnumNameActivationFunctionType(ActivationFunctionType e)
  {
    return neurun_tflite::EnumNameActivationFunctionType(e);
  }
  static const char *EnumNameTensorType(TensorType e)
  {
    return neurun_tflite::EnumNameTensorType(e);
  }
  static const Model *GetModel(const void *buf) { return neurun_tflite::GetModel(buf); }
  static bool VerifyModelBuffer(Verifier &verifier)
  {
    return neurun_tflite::VerifyModelBuffer(verifier);
  }
};

class TFLiteLoader final : public base_loader::BaseLoader<LoaderDomain, TFLiteLoader>
{
public:
  using BaseLoader::BaseLoader;

  void loadSubgraph(const neurun_tflite::SubGraph *subgraph)
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
} // namespace neurun
