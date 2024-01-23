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

#include "loader/tflite_loader.h"

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
  using Metadata = onert_tflite::Metadata;
  using Operator = onert_tflite::Operator;
  using Padding = onert_tflite::Padding;
  using Pool2DOptions = onert_tflite::Pool2DOptions;
  using Tensor = onert_tflite::Tensor;
  using TensorType = onert_tflite::TensorType;
  using SubGraph = onert_tflite::SubGraph;
  using DimensionType = onert_tflite::DimensionType;
  using SparseIndexVector = onert_tflite::SparseIndexVector;

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

class TFLiteLoader final : public base_loader::BaseLoader<LoaderDomain>
{
protected:
  // Different option name
  //  Circle: adjoint_lhs, adjoint_rhs
  //  TFLite: adj_x, adj_y
  void loadBatchMatMul(const Operator *op, ir::Graph &subg);

public:
  using BaseLoader::BaseLoader;

  bool allowOptionalInputTensor(BuiltinOperator op) override
  {
    switch (op)
    {
      case BuiltinOperator::BuiltinOperator_FULLY_CONNECTED:
      case BuiltinOperator::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
        return true;
      default:
        return false;
    }
  }

private:
  std::unique_ptr<ir::Graph> loadSubgraph(const onert_tflite::SubGraph *tflite_subg) override
  {
    auto subg = std::make_unique<ir::Graph>();
    // Load tensors
    _tensor_to_operand.resize(tflite_subg->tensors()->size());
    for (flatbuffers::uoffset_t i = 0; i < tflite_subg->tensors()->size(); ++i)
    {
      _tensor_to_operand[i] = loadOperand(tflite_subg->tensors()->Get(i), *subg);
    }
    // Set inputs
    for (const std::int32_t input_ind : *tflite_subg->inputs())
    {
      subg->addInput(tensorIdxToOperandIdx(input_ind),
                     _tensor_names.at(_tensor_to_operand[input_ind]));
    }
    // Set outputs
    for (const std::int32_t output_ind : *tflite_subg->outputs())
    {
      subg->addOutput(tensorIdxToOperandIdx(output_ind),
                      _tensor_names.at(_tensor_to_operand[output_ind]));
    }
    // Create operations
    for (const auto *op : *tflite_subg->operators())
    {
      loadOperation(op, *subg);
    }

    subg->verify();

    return subg;
  }

  void loadOperation(const onert_tflite::Operator *op, ir::Graph &subg)
  {
    auto const builtin_op = getBuiltinOperator(op);

    switch (builtin_op)
    {
      case onert_tflite::BuiltinOperator::BuiltinOperator_BATCH_MATMUL:
        loadBatchMatMul(op, subg);
        return;
      default:
        BaseLoader::loadOperation(op, subg);
        return;
    }
  }
};

void TFLiteLoader::loadBatchMatMul(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::BatchMatMul::Param param;
  const auto *options = op->builtin_options_as_BatchMatMulOptions();

  param.adj_x = options->adj_x();
  param.adj_y = options->adj_y();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::BatchMatMul(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

} // namespace

std::unique_ptr<ir::Model> loadModel(const std::string &filename)
{
  auto model = std::make_unique<ir::Model>();
  TFLiteLoader loader(model);
  loader.loadFromFile(filename);
  return model;
}

} // namespace tflite_loader
} // namespace onert
