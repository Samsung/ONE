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
#include "traininfo_loader.h"

namespace onert
{
namespace circle_loader
{

namespace
{

using loadMetadata_t = std::unique_ptr<ir::IMetadata> (*)(const uint8_t *, const size_t);

static const std::unordered_map<std::string, loadMetadata_t> metadata_loaders = {
#ifdef ONERT_TRAIN
  {"CIRCLE_TRAINING", &traininfo_loader::loadTrainingInfo}
#endif // ONERT_TRAIN
  // If necessary, add other metadata loader here
};

struct LoaderDomain
{
  using Verifier = flatbuffers::Verifier;
  using ActivationFunctionType = circle::ActivationFunctionType;
  using Buffer = circle::Buffer;
  using BuiltinOperator = circle::BuiltinOperator;
  using CustomOptionsFormat = circle::CustomOptionsFormat;
  using Metadata = circle::Metadata;
  using Model = circle::Model;
  using Operator = circle::Operator;
  using Padding = circle::Padding;
  using Pool2DOptions = circle::Pool2DOptions;
  using Tensor = circle::Tensor;
  using TensorType = circle::TensorType;
  using SubGraph = circle::SubGraph;
  using DimensionType = circle::DimensionType;
  using SparseIndexVector = circle::SparseIndexVector;

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

class CircleLoader final : public base_loader::BaseLoader<LoaderDomain>
{
protected:
  // Different option name
  //  Circle: adjoint_lhs, adjoint_rhs
  //  TFLite: adj_x, adj_y
  void loadBatchMatMul(const Operator *op, ir::Graph &subg);

  // Only circle operations
  void loadInstanceNorm(const Operator *op, ir::Graph &subg);
  void loadBCQFullyConnected(const Operator *op, ir::Graph &subg);
  void loadBCQGather(const Operator *op, ir::Graph &subg);

public:
  using BaseLoader::BaseLoader;

  bool allowOptionalInputTensor(BuiltinOperator op) override
  {
    switch (op)
    {
      case BuiltinOperator::BuiltinOperator_FULLY_CONNECTED:
      case BuiltinOperator::BuiltinOperator_BCQ_FULLY_CONNECTED:
      case BuiltinOperator::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
        return true;
      default:
        return false;
    }
  }

private:
  std::unique_ptr<ir::IMetadata> loadMetadata(const circle::Metadata *circle_meta) override
  {
    const auto name = circle_meta->name()->str();
    if (metadata_loaders.find(name) == metadata_loaders.end())
      return nullptr;

    const auto loader = metadata_loaders.at(name);

    const auto buffer_idx = circle_meta->buffer();
    const auto buffer = _domain_model->buffers()->Get(buffer_idx)->data();

    return loader(buffer->data(), buffer->size());
  }

  std::unique_ptr<ir::Graph> loadSubgraph(const circle::SubGraph *circle_subg) override
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
      subg->addInput(tensorIdxToOperandIdx(input_ind),
                     _tensor_names.at(_tensor_to_operand[input_ind]));
    }
    // Set outputs
    for (const std::int32_t output_ind : *circle_subg->outputs())
    {
      subg->addOutput(tensorIdxToOperandIdx(output_ind),
                      _tensor_names.at(_tensor_to_operand[output_ind]));
    }
    // Create operations
    for (const auto *op : *circle_subg->operators())
    {
      CircleLoader::loadOperation(op, *subg);
    }

    // TODO Remove frontend layout feature
    subg->setLayout(ir::Layout::NHWC);

    subg->verify();

    return subg;
  }

  void loadOperation(const circle::Operator *op, ir::Graph &subg)
  {
    auto const builtin_op = getBuiltinOperator(op);

    switch (builtin_op)
    {
      case circle::BuiltinOperator::BuiltinOperator_BATCH_MATMUL:
        loadBatchMatMul(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_INSTANCE_NORM:
        loadInstanceNorm(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_BCQ_FULLY_CONNECTED:
        loadBCQFullyConnected(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_BCQ_GATHER:
        loadBCQGather(op, subg);
        return;
      default:
        BaseLoader::loadOperation(op, subg);
        return;
    }
  }
};

void CircleLoader::loadBatchMatMul(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::BatchMatMul::Param param;
  const auto *options = op->builtin_options_as_BatchMatMulOptions();

  param.adj_x = options->adjoint_lhs();
  param.adj_y = options->adjoint_rhs();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::BatchMatMul(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadInstanceNorm(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::InstanceNorm::Param param;
  const auto *options = op->builtin_options_as_InstanceNormOptions();

  param.activation = convertActivation(options->fused_activation_function());
  // Use default value 1e-5 if value of epsilon is zero
  param.epsilon = options->epsilon() == 0.f ? 1e-5 : options->epsilon();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::InstanceNorm(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadBCQGather(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::BCQGather::Param param;
  const auto *options = op->builtin_options_as_BCQGatherOptions();
  param.input_hidden_size = options->input_hidden_size();
  param.axis = options->axis();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::BCQGather(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadBCQFullyConnected(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::BCQFullyConnected::Param param;
  const auto *options = op->builtin_options_as_BCQFullyConnectedOptions();
  param.weights_hidden_size = options->weights_hidden_size();
  param.activation = convertActivation(options->fused_activation_function());

  std::unique_ptr<ir::Operation> new_op(
    new ir::operation::BCQFullyConnected(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

} // namespace

std::unique_ptr<ir::Model> loadModel(const std::string &filename)
{
  auto model = std::make_unique<ir::Model>();
  CircleLoader loader(model);
  loader.loadFromFile(filename);
  return model;
}

std::unique_ptr<ir::Model> loadModel(uint8_t *buffer, size_t size)
{
  auto model = std::make_unique<ir::Model>();
  CircleLoader loader(model);
  loader.loadFromBuffer(buffer, size);
  return model;
}

} // namespace circle_loader
} // namespace onert
