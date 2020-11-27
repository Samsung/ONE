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

#include "caffe2_importer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2_op_types.h"
#include "caffe2_op_creator.h"
#include "caffe2_proto_helper.h"

#include "mir/ops/InputOp.h"
#include "mir/ops/OutputOp.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <fcntl.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <set>

namespace
{

using namespace mir_caffe2;

class Caffe2Importer
{
public:
  explicit Caffe2Importer(std::string predict_net, std::string init_net,
                          const std::vector<std::vector<int>> &input_shapes);

  /// @brief Load the model and convert it into a MIR Graph.
  std::unique_ptr<mir::Graph> importModel();

  ~Caffe2Importer();

private:
  std::string _predictNet;
  std::string _initNet;
  std::unique_ptr<mir::Graph> _graph;
  std::unique_ptr<caffe2::NetDef> _predict_net;
  std::unique_ptr<caffe2::NetDef> _init_net;
  std::unique_ptr<Caffe2OpCreator> _opCreator;
  std::vector<mir::Shape> _inputShapes;

  static const std::map<std::string, SupportedCaffe2OpType> _operatorTypes;

  // Maps Caffe2 operator input names to corresponding MIR operation outputs.
  std::unordered_map<std::string, mir::Operation::Output *> _blobNameToOutput;

  void import();
  std::unique_ptr<mir::Graph> createIR();

  /**
   * @brief Pass through caffe2 graph and collect ops unsupported by NNC
   * @throw PassException with message, containing detected problems
   */
  void collectUnsupportedOps();

  /**
   * @brief Creating MIR node from single caffe2 operator
   */
  void createMIRNodesFromOp(const ::caffe2::OperatorDef &op);

  /**
   * @brief Returns MIR operation outputs corresponding to the inputs of the given operator.
   */
  std::vector<mir::Operation::Output *> getInputMIROps(const ::caffe2::OperatorDef &op);

  void setOutputForTensor(const std::string &tensor_name, Operation::Output *output);
  mir::Operation::Output *getOutputForTensor(const std::string &name) const;

  /**
   * @brief Mark output MIR nodes
   */
  void setGraphOutputs();
};

using namespace ::caffe2;
using mir::Shape;

Caffe2Importer::Caffe2Importer(std::string predict_net, std::string init_net,
                               const std::vector<std::vector<int>> &input_shapes)
  : _predictNet(std::move(predict_net)), _initNet(std::move(init_net))
{
  for (auto &shape : input_shapes)
    _inputShapes.emplace_back(shape);

  _graph = std::make_unique<mir::Graph>();
  _opCreator = std::make_unique<Caffe2OpCreator>(_graph.get());
}

Caffe2Importer::~Caffe2Importer() = default;

static void loadModelFile(const std::string &filename, caffe2::NetDef *net)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int file_handle = open(filename.c_str(), O_RDONLY);

  if (file_handle == -1)
    throw std::runtime_error("Couldn't open file \"" + filename + "\": " + std::strerror(errno) +
                             ".");

  google::protobuf::io::FileInputStream file_stream(file_handle);
  file_stream.SetCloseOnDelete(true);

  google::protobuf::io::CodedInputStream coded_stream(&file_stream);
  coded_stream.SetTotalBytesLimit(INT_MAX, INT_MAX);

  if (!net->ParseFromCodedStream(&coded_stream))
    throw std::runtime_error("Couldn't parse file \"" + filename + "\".");

  // If the file has not been consumed entirely, assume that the file is in the wrong format.
  if (!coded_stream.ConsumedEntireMessage())
    throw std::runtime_error("File \"" + filename + "\" has not been consumed entirely.");
}

void Caffe2Importer::import()
{
  _predict_net = std::make_unique<NetDef>();
  loadModelFile(_predictNet, _predict_net.get());

  _init_net = std::make_unique<NetDef>();
  loadModelFile(_initNet, _init_net.get());

  collectUnsupportedOps();
}

std::unique_ptr<mir::Graph> Caffe2Importer::createIR()
{
  // Load initializers.
  for (const auto &op : _init_net->op())
    createMIRNodesFromOp(op);

  // Create inputs. This has to be done after processing initializers, because they may contain
  // fake inputs.
  // TODO Caffe2 does not provide a way to detect model inputs and outputs. For now assume that:
  //      - there is exactly one input;
  //      - the input is for the first layer;
  //      - the input has 'float' element type.
  const auto &input_name = _predict_net->op(0).input(0);
  mir::TensorType input_type(mir::DataType::FLOAT32, _inputShapes[0]);
  auto input = _graph->create<mir::ops::InputOp>(input_type)->getOutput(0);
  setOutputForTensor(input_name, input);

  for (const auto &op : _predict_net->op())
    createMIRNodesFromOp(op);

  setGraphOutputs();

  return std::move(_graph);
}

std::unique_ptr<mir::Graph> Caffe2Importer::importModel()
{
  import();
  return createIR();
}

void Caffe2Importer::collectUnsupportedOps()
{
  std::set<std::string> unsupportedOps;
  for (const auto &op : _predict_net->op())
  {
    if (_operatorTypes.find(op.type()) == _operatorTypes.end())
      unsupportedOps.insert(op.type());
  }

  if (!unsupportedOps.empty())
  {
    std::string exceptionMsg("Can't load model, unsupported operators:");
    for (const auto &op : unsupportedOps)
      exceptionMsg.append("\n  * " + op);
    throw std::runtime_error(exceptionMsg);
  }
}

void Caffe2Importer::createMIRNodesFromOp(const OperatorDef &op)
{
  std::vector<mir::Operation::Output *> outputs;

  auto inputs = getInputMIROps(op);

  SupportedCaffe2OpType opType = _operatorTypes.at(op.type());
  switch (opType)
  {
    case SupportedCaffe2OpType::constantFill:
    case SupportedCaffe2OpType::givenTensorFill:
    case SupportedCaffe2OpType::givenTensorInt64Fill:
      outputs = _opCreator->convertConstant(inputs, op);
      break;
    case SupportedCaffe2OpType::add:
      outputs = _opCreator->convertAdd(inputs, op);
      break;
    case SupportedCaffe2OpType::averagePool:
      outputs = _opCreator->convertAveragePool(inputs, op);
      break;
    case SupportedCaffe2OpType::conv:
      outputs = _opCreator->convertConv(inputs, op);
      break;
    case SupportedCaffe2OpType::concat:
      outputs = _opCreator->convertConcat(inputs, op);
      break;
    case SupportedCaffe2OpType::dropout:
      outputs = _opCreator->convertDropout(inputs, op);
      break;
    case SupportedCaffe2OpType::FC:
      outputs = _opCreator->convertFC(inputs, op);
      break;
    case SupportedCaffe2OpType::maxPool:
      outputs = _opCreator->convertMaxPool(inputs, op);
      break;
    case SupportedCaffe2OpType::mul:
      outputs = _opCreator->convertMul(inputs, op);
      break;
    case SupportedCaffe2OpType::relu:
      outputs = _opCreator->convertRelu(inputs);
      break;
    case SupportedCaffe2OpType::resizeNearest:
      outputs = _opCreator->convertResizeNearest(inputs, op);
      break;
    case SupportedCaffe2OpType::sigmoid:
      outputs = _opCreator->convertSigmoid(inputs);
      break;
    case SupportedCaffe2OpType::softmax:
      outputs = _opCreator->convertSoftmax(inputs, op);
      break;
    case SupportedCaffe2OpType::spatialBN:
      outputs = _opCreator->convertSpatialBN(inputs, op);
      break;
    case SupportedCaffe2OpType::sum:
      outputs = _opCreator->convertSum(inputs);
      break;
    case SupportedCaffe2OpType::clip:
      outputs = _opCreator->convertClip(inputs, op);
      break;
    case SupportedCaffe2OpType::reshape:
      outputs = _opCreator->convertReshape(inputs, op);
      break;
    default:
      assert(false && "All unsupported types should have been found before this pass.");
  }

  for (size_t i = 0; i < outputs.size(); ++i)
  {
    setOutputForTensor(op.output(i), outputs[i]);
  }
}

std::vector<mir::Operation::Output *> Caffe2Importer::getInputMIROps(const OperatorDef &op)
{
  std::vector<mir::Operation::Output *> inputs;

  for (const auto &input_name : op.input())
  {
    inputs.push_back(getOutputForTensor(input_name));
  }

  return inputs;
}

void Caffe2Importer::setOutputForTensor(const std::string &tensor_name, Operation::Output *output)
{
  auto it = _blobNameToOutput.find(tensor_name);
  if (it != _blobNameToOutput.cend())
  {
    // caffe2 input blob name could be same as output blob name, and next line will overwrite
    // '_blobNameToOpOutput' element, but in all networks that I saw it was not a problem
    it->second->setName("");
  }
  output->setName(tensor_name);
  _blobNameToOutput[tensor_name] = output;
}

mir::Operation::Output *Caffe2Importer::getOutputForTensor(const std::string &name) const
{
  return _blobNameToOutput.at(name);
}

void Caffe2Importer::setGraphOutputs()
{
  // Create outputs.
  // TODO Caffe2 does not provide a way to detect model inputs and outputs. For now assume that:
  //      - there is exactly one output;
  //      - the output is from the last layer.
  const auto &output_name = _predict_net->op().rbegin()->output(0);
  auto output = getOutputForTensor(output_name);
  _graph->create<mir::ops::OutputOp>(output);
}

const std::map<std::string, SupportedCaffe2OpType> Caffe2Importer::_operatorTypes = {
  {"Add", SupportedCaffe2OpType::add},
  {"AveragePool", SupportedCaffe2OpType::averagePool},
  {"Conv", SupportedCaffe2OpType::conv},
  {"Concat", SupportedCaffe2OpType::concat},
  {"ConstantFill", SupportedCaffe2OpType::constantFill},
  {"Dropout", SupportedCaffe2OpType::dropout},
  {"FC", SupportedCaffe2OpType::FC},
  {"GivenTensorFill", SupportedCaffe2OpType::givenTensorFill},
  {"MaxPool", SupportedCaffe2OpType::maxPool},
  {"Mul", SupportedCaffe2OpType::mul},
  {"Relu", SupportedCaffe2OpType::relu},
  {"ResizeNearest", SupportedCaffe2OpType::resizeNearest},
  {"Sigmoid", SupportedCaffe2OpType::sigmoid},
  {"Softmax", SupportedCaffe2OpType::softmax},
  {"SpatialBN", SupportedCaffe2OpType::spatialBN},
  {"Sum", SupportedCaffe2OpType::sum},
  {"Clip", SupportedCaffe2OpType::clip},
  {"Reshape", SupportedCaffe2OpType::reshape},
  {"GivenTensorInt64Fill", SupportedCaffe2OpType::givenTensorInt64Fill},
};
}

namespace mir_caffe2
{

std::unique_ptr<mir::Graph> loadModel(std::string predict_net, std::string init_net,
                                      const std::vector<std::vector<int>> &input_shapes)
{
  Caffe2Importer importer(std::move(predict_net), std::move(init_net), input_shapes);
  return importer.importModel();
}

} // namespace mir_caffe2
