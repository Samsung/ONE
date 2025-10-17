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

#include "loader/CircleLoader.h"
#include "loader/ModelLoader.h"

#include "BaseLoader.h"
#include "circle_schema_generated.h"

#include <filesystem>

namespace onert::loader
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

class CircleLoader final : public loader::BaseLoader<LoaderDomain>
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
  void loadRmsNorm(const Operator *op, ir::Graph &subg);
  void loadRoPE(const Operator *op, ir::Graph &subg);
  void loadCall(const Operator *op, ir::Graph &subg);
  void loadRunModel(const Operator *op, ir::Graph &subg);
  void loadBCQUnembedding(const Operator *op, ir::Graph &subg);
  void loadCustom(const Operator *op, ir::Graph &subg);

public:
  using BaseLoader::BaseLoader;

  bool allowOptionalInputTensor(BuiltinOperator op) override
  {
    switch (op)
    {
      case BuiltinOperator::BuiltinOperator_FULLY_CONNECTED:
      case BuiltinOperator::BuiltinOperator_BCQ_FULLY_CONNECTED:
      case BuiltinOperator::BuiltinOperator_CUSTOM:
      case BuiltinOperator::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
        return true;
      default:
        return false;
    }
  }

protected:
  ir::DataType tensorTypeToDataType(const TensorType type) override
  {
    if (type == TensorType::TensorType_GGML_Q4_0)
      return ir::DataType::QUANT_GGML_Q4_0;
    if (type == TensorType::TensorType_GGML_Q8_0)
      return ir::DataType::QUANT_GGML_Q8_0;

    return BaseLoader::tensorTypeToDataType(type);
  }

  ir::operation::RoPE::RoPEMode convertRoPEMode(const circle::RoPEMode mode)
  {
    switch (mode)
    {
      case circle::RoPEMode::RoPEMode_GPT_NEOX:
        return ir::operation::RoPE::RoPEMode::GPT_NEOX;
      case circle::RoPEMode::RoPEMode_GPT_J:
        return ir::operation::RoPE::RoPEMode::GPT_J;
      default:
        throw std::runtime_error(std::string("Unsupported RoPE mode: ") +
                                 std::to_string(static_cast<int>(mode)));
    }
  }

private:
  std::unique_ptr<ir::Graph> loadSubgraph(const circle::SubGraph *circle_subg) override
  {
    auto subg = std::make_unique<ir::Graph>();
    // Load tensors
    _tensor_to_operand.resize(circle_subg->tensors()->size());
    for (flatbuffers::uoffset_t i = 0; i < circle_subg->tensors()->size(); ++i)
    {
      _tensor_to_operand[i] = loadOperand(circle_subg->tensors()->Get(i), *subg);
      subg->operands().at(_tensor_to_operand[i]).setOriginIndex(ir::OriginIndex(i));
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
      case circle::BuiltinOperator::BuiltinOperator_RMS_NORM:
        loadRmsNorm(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_ROPE:
        loadRoPE(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_CALL:
        loadCall(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_RUN_MODEL:
        loadRunModel(op, subg);
        return;
      case circle::BuiltinOperator::BuiltinOperator_CUSTOM:
        loadCustom(op, subg);
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

void CircleLoader::loadRmsNorm(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::RmsNorm::Param param;
  const auto *options = op->builtin_options_as_RmsNormOptions();

  // Use default value 1e-6 if value of epsilon is zero
  param.epsilon = options->epsilon() == 0.f ? 1e-6 : options->epsilon();

  std::unique_ptr<ir::Operation> new_op(new ir::operation::RmsNorm(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadRoPE(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::RoPE::Param param;
  const auto *options = op->builtin_options_as_RoPEOptions();

  param.mode = convertRoPEMode(options->mode());

  std::unique_ptr<ir::Operation> new_op(new ir::operation::RoPE(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadCall(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::Call::Param param;
  const auto *options = op->builtin_options_as_CallOptions();
  const uint32_t callee_index = options->subgraph();
  verifySubgraphIndex(callee_index);
  param.callee_subg_index = ir::SubgraphIndex{static_cast<uint16_t>(callee_index)};

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Call(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadRunModel(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  // As workaround, we assume that only tvn type model is used for RUN_MODEL
  // So subgraph will have only one OP: Bulk
  // We copy this OP into this subgraph (like model inlining)
  //
  // TODO Introduce RunModel operator type to support various model
  // TODO Introduce ModelIndexManager to handle various case
  //      - ModelIndex mapping for nnpackage's multimodel
  //      - ModelIndex mapping for RUN_MODEL
  //      - Lazy loading for model index for RUN_MODEL
  //      - Model inlining on optimize phase (compile)
  assert(!_file_path.empty());
  auto model_base_path = std::filesystem::path(_file_path).parent_path();
  auto *options = op->builtin_options_as_RunModelOptions();
  auto location = options->location()->str();
  auto model_path = model_base_path / location;
  auto extension_path = model_path.extension();
  if (extension_path.empty())
  {
    throw std::runtime_error("Model path has no extension: " + model_path.string());
  }
  auto extension = extension_path.string();
  auto type = extension.substr(1); // remove dot

  auto model = onert::loader::loadModel(model_path.string(), type);
  assert(model);

  // Get 1st OP in 1st subgraph
  auto &op_bulk = model->primary_subgraph()->operations().at(ir::OperationIndex{0});
  // Check if it is Bulk operation
  assert(op_bulk.opcode() == ir::OpCode::Bulk);
  const auto bulk_op = dynamic_cast<const ir::operation::Bulk *>(&op_bulk);

  std::unique_ptr<ir::Operation> new_op(new ir::operation::Bulk(inputs, outputs, bulk_op->param()));
  subg.addOperation(std::move(new_op));
}

void CircleLoader::loadBCQUnembedding(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  ir::operation::BCQUnembedding::Param param;
  if (op->custom_options() == nullptr)
  {
    throw std::runtime_error{"BCQUnembedding: empty option"};
  }
  else
  {
    const auto attr_map = getCustomOpAttrMap(op);
    param.weights_hidden_size = attr_map["weights_hidden_size"].AsUInt32();
    param.lsh_type = attr_map["lsh_type"].AsString().str();
    param.lsh_choices = attr_map["lsh_choices"].AsInt32();
  }

  const auto fbn = loadOperationTo<ir::operation::BCQUnembedding>(op, subg, param);

  if (fbn->getInputs().size() != 5)
  {
    throw std::runtime_error{"BCQUnembedding: NYI input - only support five inputs"};
  }
}

void CircleLoader::loadCustom(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  assert(op->custom_options_format() == CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS &&
         "Unsupported custom operation options format");

  auto *op_code = _domain_model->operator_codes()->Get(op->opcode_index());
  auto custom_op_name = op_code->custom_code()->str();

  enum class BuiltinOP
  {
    BCQUnembedding,
  };

  // Mapping from custom op name string to BuiltinOP enum
  std::map<std::string, BuiltinOP> builtin_map = {
    {"BCQUnembedding", BuiltinOP::BCQUnembedding},
  };

  // Throw out_of_range if it is unknown custom op
  auto custom_op_id = builtin_map.at(custom_op_name);
  switch (custom_op_id)
  {
    case BuiltinOP::BCQUnembedding:
      loadBCQUnembedding(op, subg);
      break;
    default:
      BaseLoader::loadOperation(op, subg);
      return;
  }

  return;
}

} // namespace

std::unique_ptr<ir::Model> loadCircleModel(const std::string &filename)
{
  return CircleLoader().loadFromFile(filename);
}

std::unique_ptr<ir::Model> loadCircleModel(uint8_t *buffer, size_t size)
{
  return CircleLoader().loadFromBuffer(buffer, size);
}

} // namespace onert::loader
