/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflite_importer.h"
#include "tflite_op_creator.h"
#include "schema_generated.h"

#include "mir/TensorVariant.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/OutputOp.h"

#include <fstream>
#include <memory>
#include <utility>
#include <vector>
#include <set>

namespace mir_tflite
{

namespace
{

class TfliteImporter
{
public:
  explicit TfliteImporter(std::string filename);

  /// @brief Load the model and convert it into a MIR Graph.
  std::unique_ptr<mir::Graph> importModel();

  ~TfliteImporter();

private:
  std::string _filename;
  std::unique_ptr<tflite::ModelT> _model;

  std::unique_ptr<mir::Graph> _graph;
  std::unique_ptr<TFLiteOpCreator> _opCreator;

  // Maps TFLite tensors indices to corresponding MIR operation outputs.
  std::vector<mir::Operation::Output *> _tensorMap;

  void import();

  void walkModel(const tflite::ModelT *model);

  void walkSubgraph(const tflite::SubGraphT *subgraph);

  void walkOperator(const tflite::SubGraphT *subgraph, const tflite::OperatorT *op);

  /**
   * @brief Pass through tflite graph and collect operators unsupported by NNC
   * @throw PassException with message, containing detected problems
   */
  void collectUnsupportedOps();

  /**
   * @brief Returns MIR operation outputs corresponding to the inputs of the given operator.
   */
  std::vector<mir::Operation::Output *> getMIRInputsForOperator(const tflite::SubGraphT *subgraph,
                                                                const tflite::OperatorT *op);
};

TfliteImporter::TfliteImporter(std::string filename) : _filename(std::move(filename))
{
  _graph = std::make_unique<mir::Graph>();
  _opCreator = std::make_unique<TFLiteOpCreator>(_graph.get());
}

TfliteImporter::~TfliteImporter() = default;

void TfliteImporter::import()
{
  std::ifstream stream(_filename, std::ios::in | std::ios::binary);
  if (stream.fail())
    throw std::runtime_error("Couldn't open file \"" + _filename + "\".");

  std::vector<char> model_buffer((std::istreambuf_iterator<char>(stream)),
                                 std::istreambuf_iterator<char>());

  if (stream.fail())
    throw std::runtime_error("Couldn't read file \"" + _filename + "\".");

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t *>(model_buffer.data()),
                                 model_buffer.size());

  if (!tflite::VerifyModelBuffer(verifier))
    throw std::runtime_error("Could not load model: " + _filename + "\n");

  _model = tflite::UnPackModel(model_buffer.data());
}

static const std::set<tflite::BuiltinOperator> supportedOperators = {
  tflite::BuiltinOperator_ADD,
  tflite::BuiltinOperator_AVERAGE_POOL_2D,
  tflite::BuiltinOperator_CONCATENATION,
  tflite::BuiltinOperator_CONV_2D,
  tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  tflite::BuiltinOperator_DIV,
  tflite::BuiltinOperator_FULLY_CONNECTED,
  tflite::BuiltinOperator_HARD_SWISH,
  tflite::BuiltinOperator_LEAKY_RELU,
  tflite::BuiltinOperator_LOGISTIC,
  tflite::BuiltinOperator_MAX_POOL_2D,
  tflite::BuiltinOperator_MAXIMUM,
  tflite::BuiltinOperator_MEAN,
  tflite::BuiltinOperator_MUL,
  tflite::BuiltinOperator_PAD,
  tflite::BuiltinOperator_RELU,
  tflite::BuiltinOperator_RELU6,
  tflite::BuiltinOperator_RESHAPE,
  tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
  tflite::BuiltinOperator_RSQRT,
  tflite::BuiltinOperator_SHAPE,
  tflite::BuiltinOperator_SLICE,
  tflite::BuiltinOperator_SOFTMAX,
  tflite::BuiltinOperator_SQRT,
  tflite::BuiltinOperator_SQUARED_DIFFERENCE,
  tflite::BuiltinOperator_SQUEEZE,
  tflite::BuiltinOperator_STRIDED_SLICE,
  tflite::BuiltinOperator_SUB,
  tflite::BuiltinOperator_TANH,
  tflite::BuiltinOperator_TRANSPOSE,
  tflite::BuiltinOperator_TRANSPOSE_CONV,
};

void TfliteImporter::collectUnsupportedOps()
{
  std::set<std::string> errors;
  for (const auto &subgraph : _model->subgraphs)
    for (const auto &op : subgraph->operators)
    {
      tflite::BuiltinOperator opcode = _model->operator_codes[op->opcode_index]->builtin_code;
      if (supportedOperators.find(opcode) == supportedOperators.end())
      {
        if (opcode <= tflite::BuiltinOperator_MAX)
          errors.insert(std::string(EnumNameBuiltinOperator(opcode)) + ": unsupported operator");
        else
          errors.insert(std::to_string(opcode) + ": unsuppored in tflite custom opcode");
      }
    }

  if (!errors.empty())
  {
    std::string msg("NNC can't load model. Detected problems:");
    for (const auto &e : errors)
      msg.append("\n  * " + e);
    throw std::runtime_error(msg);
  }
}

std::unique_ptr<mir::Graph> TfliteImporter::importModel()
{
  import();
  collectUnsupportedOps();
  walkModel(_model.get());
  return std::move(_graph);
}

void TfliteImporter::walkModel(const tflite::ModelT *model)
{
  for (const auto &subgraph : model->subgraphs)
    walkSubgraph(subgraph.get());
}

mir::DataType convertElementType(tflite::TensorType type)
{
  switch (type)
  {
    case tflite::TensorType_INT32:
      return mir::DataType::INT32;
    case tflite::TensorType_FLOAT32:
      return mir::DataType::FLOAT32;
    case tflite::TensorType_INT64:
      return mir::DataType::INT64;
    case tflite::TensorType_UINT8:
      return mir::DataType::UINT8;
    default:
      throw std::runtime_error(std::string("Unsupported tensor type: ") + EnumNameTensorType(type));
  }
}

mir::TensorType getMirTensorType(const tflite::TensorT &tensor)
{
  mir::DataType element_type = convertElementType(tensor.type);

  mir::Shape shape(tensor.shape.size());
  for (std::size_t i = 0; i < tensor.shape.size(); ++i)
  {
    shape.dim(i) = tensor.shape[i];
  }

  if (tensor.quantization != nullptr)
  {
    const tflite::QuantizationParametersT &params = *tensor.quantization;

    if (params.details.type != tflite::QuantizationDetails_NONE)
      throw std::runtime_error("Custom quantization is not supported.");

    // Empty parameters mean no quantization at all.
    if (params.scale.empty() && params.zero_point.empty())
      return mir::TensorType{element_type, shape};

    if (params.scale.size() != 1 || params.zero_point.size() != 1)
      throw std::runtime_error("Non-scalar quantization is not supported.");

    mir::AffineQuantization quantization{params.scale[0], static_cast<int>(params.zero_point[0])};

    return mir::TensorType{element_type, shape, quantization};
  }
  else
  {
    return mir::TensorType{element_type, shape};
  }
}

void TfliteImporter::walkSubgraph(const tflite::SubGraphT *subgraph)
{
  _tensorMap.assign(subgraph->tensors.size(), nullptr);

  for (const auto input_tensor_index : subgraph->inputs)
  {
    const tflite::TensorT &tensor = *subgraph->tensors[input_tensor_index];

    mir::TensorType input_type = getMirTensorType(tensor);
    auto input = _graph->create<mir::ops::InputOp>(input_type)->getOutput(0);
    input->setName(tensor.name);

    assert(_tensorMap[input_tensor_index] == nullptr);
    _tensorMap[input_tensor_index] = input;
  }

  for (const auto &op : subgraph->operators)
  {
    walkOperator(subgraph, op.get());
  }

  for (const auto output_tensor_index : subgraph->outputs)
  {
    auto output = _tensorMap[output_tensor_index];
    _graph->create<mir::ops::OutputOp>(output);
  }
}

void TfliteImporter::walkOperator(const tflite::SubGraphT *subgraph, const tflite::OperatorT *op)
{
  std::vector<mir::Operation::Output *> inputs = getMIRInputsForOperator(subgraph, op);
  std::vector<mir::Operation::Output *> outputs;

  tflite::BuiltinOperator opcode = _model->operator_codes[op->opcode_index]->builtin_code;
  switch (opcode)
  {
    case tflite::BuiltinOperator_CONV_2D:
      outputs = _opCreator->convertConv2D(op->builtin_options.AsConv2DOptions(), inputs);
      break;
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
      outputs =
        _opCreator->convertDepthwiseConv2D(op->builtin_options.AsDepthwiseConv2DOptions(), inputs);
      break;
    case tflite::BuiltinOperator_MAX_POOL_2D:
      outputs = _opCreator->convertMaxPool2D(op->builtin_options.AsPool2DOptions(), inputs);
      break;
    case tflite::BuiltinOperator_AVERAGE_POOL_2D:
      outputs = _opCreator->convertAveragePool2D(op->builtin_options.AsPool2DOptions(), inputs);
      break;
    case tflite::BuiltinOperator_CONCATENATION:
      outputs =
        _opCreator->convertConcatenation(op->builtin_options.AsConcatenationOptions(), inputs);
      break;
    case tflite::BuiltinOperator_RESHAPE:
      outputs = _opCreator->convertReshape(op->builtin_options.AsReshapeOptions(), inputs);
      break;
    case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      outputs = _opCreator->convertResizeNearestNeighbor(
        op->builtin_options.AsResizeNearestNeighborOptions(), inputs);
      break;
    case tflite::BuiltinOperator_MEAN:
      outputs = _opCreator->convertMean(op->builtin_options.AsReducerOptions(), inputs);
      break;
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      outputs =
        _opCreator->convertFullyConnected(op->builtin_options.AsFullyConnectedOptions(), inputs);
      break;
    case tflite::BuiltinOperator_SOFTMAX:
      outputs = _opCreator->convertSoftmax(op->builtin_options.AsSoftmaxOptions(), inputs);
      break;
    case tflite::BuiltinOperator_SLICE:
      outputs = _opCreator->convertSlice(op->builtin_options.AsSliceOptions(), inputs);
      break;
    case tflite::BuiltinOperator_SQUEEZE:
      outputs = _opCreator->convertSqueeze(op->builtin_options.AsSqueezeOptions(), inputs);
      break;
    case tflite::BuiltinOperator_LOGISTIC:
      outputs = _opCreator->convertLogistic(inputs);
      break;
    case tflite::BuiltinOperator_RSQRT:
      outputs = _opCreator->convertRsqrt(inputs);
      break;
    case tflite::BuiltinOperator_SQRT:
      outputs = _opCreator->convertSqrt(inputs);
      break;
    case tflite::BuiltinOperator_ADD:
      outputs = _opCreator->convertAdd(op->builtin_options.AsAddOptions(), inputs);
      break;
    case tflite::BuiltinOperator_SUB:
      outputs = _opCreator->convertSub(op->builtin_options.AsSubOptions(), inputs);
      break;
    case tflite::BuiltinOperator_MUL:
      outputs = _opCreator->convertMul(op->builtin_options.AsMulOptions(), inputs);
      break;
    case tflite::BuiltinOperator_DIV:
      outputs = _opCreator->convertDiv(op->builtin_options.AsDivOptions(), inputs);
      break;
    case tflite::BuiltinOperator_MAXIMUM:
      outputs = _opCreator->convertMax(inputs);
      break;
    case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
      outputs = _opCreator->convertSquaredDifference(inputs);
      break;
    case tflite::BuiltinOperator_TRANSPOSE_CONV:
      outputs =
        _opCreator->convertTransposeConv(op->builtin_options.AsTransposeConvOptions(), inputs);
      break;
    case tflite::BuiltinOperator_PAD:
      outputs = _opCreator->convertPad(op->builtin_options.AsPadOptions(), inputs);
      break;
    case tflite::BuiltinOperator_TANH:
      outputs = _opCreator->convertTanh(inputs);
      break;
    case tflite::BuiltinOperator_RELU:
      outputs = _opCreator->convertReLU(inputs);
      break;
    case tflite::BuiltinOperator_RELU6:
      outputs = _opCreator->convertReLU6(inputs);
      break;
    case tflite::BuiltinOperator_TRANSPOSE:
      outputs = _opCreator->convertTranspose(op->builtin_options.AsTransposeOptions(), inputs);
      break;
    case tflite::BuiltinOperator_STRIDED_SLICE:
      outputs =
        _opCreator->convertStridedSlice(op->builtin_options.AsStridedSliceOptions(), inputs);
      break;
    case tflite::BuiltinOperator_LEAKY_RELU:
      outputs = _opCreator->convertLeakyReLU(op->builtin_options.AsLeakyReluOptions(), inputs);
      break;
    case tflite::BuiltinOperator_SHAPE:
      outputs = _opCreator->convertShape(op->builtin_options.AsShapeOptions(), inputs);
      break;
    case tflite::BuiltinOperator_HARD_SWISH:
      outputs = _opCreator->convertHardSwish(op->builtin_options.AsHardSwishOptions(), inputs);
      break;
    default:
      assert(false && "All unsupported types should have been found before this pass.");
  }

  assert(outputs.size() == op->outputs.size());
  for (std::size_t i = 0; i < op->outputs.size(); ++i)
  {
    const auto tensor_index = op->outputs[i];
    const tflite::TensorT &tensor = *subgraph->tensors[tensor_index];

    mir::TensorType output_type = getMirTensorType(tensor);

    // The type should have been inferred correctly, except for quantization information.
    assert(outputs[i]->getType().getElementType() == output_type.getElementType() &&
           outputs[i]->getType().getShape() == output_type.getShape());

    outputs[i]->setName(tensor.name);
    outputs[i]->setType(output_type);

    assert(_tensorMap[tensor_index] == nullptr);
    _tensorMap[tensor_index] = outputs[i];
  }
}

std::vector<mir::Operation::Output *>
TfliteImporter::getMIRInputsForOperator(const tflite::SubGraphT *subgraph,
                                        const tflite::OperatorT *op)
{
  std::vector<mir::Operation::Output *> inputs;

  for (const auto tensor_index : op->inputs)
  {
    const tflite::TensorT &tensor = *subgraph->tensors[tensor_index];
    const tflite::BufferT &buffer = *_model->buffers[tensor.buffer];
    if (!buffer.data.empty())
    {
      assert(_tensorMap[tensor_index] == nullptr);
      mir::TensorType type = getMirTensorType(tensor);
      mir::TensorVariant mir_tensor{type, buffer.data.data()};
      inputs.emplace_back(_graph->create<mir::ops::ConstantOp>(mir_tensor)->getOutput(0));
    }
    else
    {
      assert(_tensorMap[tensor_index] != nullptr);
      // By this point every input for the operation "op" should have corresponding
      // Model IR operations that output its inputs. This assumption is provided by the fact
      // that TFLite format specifies all operations in the execution order.
      inputs.emplace_back(_tensorMap[tensor_index]);
    }
  }

  return inputs;
}

} // namespace

std::unique_ptr<mir::Graph> loadModel(std::string filename)
{
  TfliteImporter importer(std::move(filename));
  return importer.importModel();
}

} // namespace mir_tflite
