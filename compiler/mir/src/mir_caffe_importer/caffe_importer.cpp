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

#include "caffe_importer.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe_op_creator.h"
#include "caffe_op_types.h"

#include "mir/ops/OutputOp.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>

#include <fcntl.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <set>

namespace mir_caffe
{

namespace
{

class CaffeImporter
{
public:
  /// @brief Load the model and convert it into a MIR Graph.
  std::unique_ptr<mir::Graph> importModelFromBinaryFile(const std::string &filename);
  std::unique_ptr<mir::Graph> importModelFromTextFile(const std::string &filename);

private:
  std::unique_ptr<mir::Graph> importModel();

  std::unique_ptr<caffe::NetParameter> _net;
  std::unique_ptr<CaffeOpCreator> _opCreator;

  // Maps Caffe blob names to corresponding MIR operation outputs.
  std::map<std::string, mir::Operation::Output *> _blobNameToOpOutput;

  static const std::map<std::string, CaffeOpType> _operatorTypes;

  /**
   * @brief Mark output MIR nodes
   */
  void setGraphOutputs(mir::Graph *graph);

  /**
   * @brief Pass through caffe graph and collect unsupported by NNC layers
   * @throw PassException with message, containing detected problems
   */
  void collectUnsupportedLayers();

  /**
   * @brief Create MIR node from single caffe layer
   */
  void createMIRNodesFromLayer(const caffe::LayerParameter &layer);

  mir::Operation::Output *getOutputForBlob(const std::string &blob_name) const;
  void setOutputForBlob(const std::string &blob_name, mir::Operation::Output *output);

  /**
   * @brief Collect unsupported parts of caffe layer
   */
  void collectUnsupportedOp(const caffe::LayerParameter &layer, std::set<std::string> &problems);

  /**
   * @brief Returns MIR operation outputs corresponding to the inputs of the given layer.
   */
  std::vector<mir::Operation::Output *> getMIRInputsForLayer(const caffe::LayerParameter &layer);

  void processDeprecatedInput();
};

void loadModelFromBinaryFile(const std::string &filename, caffe::NetParameter *net)
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

void loadModelFromTextFile(const std::string &filename, caffe::NetParameter *net)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int file_handle = open(filename.c_str(), O_RDONLY);

  if (file_handle == -1)
    throw std::runtime_error("Couldn't open file \"" + filename + "\": " + std::strerror(errno) +
                             ".");

  google::protobuf::io::FileInputStream file_stream(file_handle);
  file_stream.SetCloseOnDelete(true);

  if (!google::protobuf::TextFormat::Parse(&file_stream, net))
    throw std::runtime_error("Couldn't parse file \"" + filename + "\".");
}

std::unique_ptr<mir::Graph> CaffeImporter::importModel()
{
  auto graph = std::make_unique<mir::Graph>();
  _opCreator = std::make_unique<CaffeOpCreator>(graph.get());

  collectUnsupportedLayers();

  for (int i = 0; i < _net->layer_size(); ++i)
    createMIRNodesFromLayer(_net->layer(i));

  setGraphOutputs(graph.get());

  return graph;
}

std::unique_ptr<mir::Graph> CaffeImporter::importModelFromBinaryFile(const std::string &filename)
{
  _net = std::make_unique<caffe::NetParameter>();
  loadModelFromBinaryFile(filename, _net.get());

  return importModel();
}

std::unique_ptr<mir::Graph> CaffeImporter::importModelFromTextFile(const std::string &filename)
{
  _net = std::make_unique<caffe::NetParameter>();
  loadModelFromTextFile(filename, _net.get());

  return importModel();
}

void CaffeImporter::collectUnsupportedLayers()
{
  processDeprecatedInput();

  std::set<std::string> problems;

  for (const caffe::LayerParameter &layer : _net->layer())
    collectUnsupportedOp(layer, problems);

  if (!problems.empty())
  {
    std::string msg("NNC can't load model. Detected problems:");
    for (const auto &problemStr : problems)
      msg.append("\n  * " + problemStr);
    throw std::runtime_error(msg);
  }
}

void CaffeImporter::createMIRNodesFromLayer(const caffe::LayerParameter &layer)
{
  std::vector<mir::Operation::Output *> inputs = getMIRInputsForLayer(layer);
  std::vector<mir::Operation::Output *> outputs;

  switch (_operatorTypes.at(layer.type()))
  {
    case CaffeOpType::input:
      outputs = _opCreator->convertInput(layer);
      break;
    case CaffeOpType::convolution:
      outputs = _opCreator->convertConvolution(layer, inputs);
      break;
    case CaffeOpType::innerProduct:
      outputs = _opCreator->convertInnerProduct(layer, inputs);
      break;
    case CaffeOpType::pooling:
      outputs = _opCreator->convertPooling(layer, inputs);
      break;
    case CaffeOpType::concat:
      outputs = _opCreator->convertConcat(layer, inputs);
      break;
    case CaffeOpType::reshape:
      outputs = _opCreator->convertReshape(layer, inputs);
      break;
    case CaffeOpType::ReLU:
      outputs = _opCreator->convertReLU(layer, inputs);
      break;
    case CaffeOpType::softmax:
      outputs = _opCreator->convertSoftmax(layer, inputs);
      break;
    case CaffeOpType::scale:
      outputs = _opCreator->convertScale(layer, inputs);
      break;
    case CaffeOpType::batchNorm:
      outputs = _opCreator->convertBatchNorm(layer, inputs);
      break;
    case CaffeOpType::dropout:
      outputs = _opCreator->convertDropout(layer, inputs);
      break;
    case CaffeOpType::tanh:
      outputs = _opCreator->convertTanH(layer, inputs);
      break;
    case CaffeOpType::ELU:
      outputs = _opCreator->convertELU(layer, inputs);
      break;
    case CaffeOpType::eltwise:
      outputs = _opCreator->convertEltwise(layer, inputs);
      break;
    case CaffeOpType::embed:
      outputs = _opCreator->convertEmbed(layer, inputs);
      break;
    case CaffeOpType::deconvolution:
      outputs = _opCreator->convertDeconvolution(layer, inputs);
      break;
    case CaffeOpType::split:
      outputs = _opCreator->convertSplit(layer, inputs);
      break;
    case CaffeOpType::sigmoid:
      outputs = _opCreator->convertSigmoid(layer, inputs);
      break;
    case CaffeOpType::LSTM:
      outputs = _opCreator->convertLSTM(layer, inputs);
      break;
    default:
      assert(false && "All unsupported types should have been found before this pass.");
  }

  assert(static_cast<int>(outputs.size()) == layer.top_size() && "Number of outputs differs.");
  for (int i = 0; i < layer.top_size(); ++i)
    setOutputForBlob(layer.top(i), outputs[i]);
}

void CaffeImporter::collectUnsupportedOp(const caffe::LayerParameter &layer,
                                         std::set<std::string> &problems)
{
  auto it = _operatorTypes.find(layer.type());
  if (it == _operatorTypes.end())
  {
    problems.insert(layer.type() + ": unknown layer");
    return;
  }

  CaffeOpType op_type = it->second;

  switch (op_type)
  {
    case CaffeOpType::concat:
    case CaffeOpType::input:
    case CaffeOpType::softmax:
    case CaffeOpType::scale:
    case CaffeOpType::dropout:
    case CaffeOpType::split:
    case CaffeOpType::eltwise:
    case CaffeOpType::ELU:
    case CaffeOpType::ReLU:
    case CaffeOpType::embed:
    case CaffeOpType::sigmoid:
    case CaffeOpType::tanh:
    case CaffeOpType::innerProduct:
      // No checks
      break;
    case CaffeOpType::deconvolution:
    case CaffeOpType::convolution:
      _opCreator->checkConvolution(layer, problems);
      break;
    case CaffeOpType::pooling:
      _opCreator->checkPooling(layer, problems);
      break;
    case CaffeOpType::reshape:
      _opCreator->checkReshape(layer, problems);
      break;
    case CaffeOpType::batchNorm:
      _opCreator->checkBatchNorm(layer, problems);
      break;
    case CaffeOpType::LSTM:
      _opCreator->checkLSTM(layer, problems);
      break;
    default:
      problems.insert(layer.type() + ": unsupported layer");
      break;
  }
}

void CaffeImporter::processDeprecatedInput()
{
  if (_net->input_dim_size() != 0 || _net->input_shape_size() != 0)
    throw std::runtime_error("Deprecated Caffe input types are not supported");
}

std::vector<mir::Operation::Output *>
CaffeImporter::getMIRInputsForLayer(const caffe::LayerParameter &layer)
{
  std::vector<mir::Operation::Output *> inputs;

  for (const auto &input_name : layer.bottom())
    inputs.push_back(getOutputForBlob(input_name));

  return inputs;
}

mir::Operation::Output *CaffeImporter::getOutputForBlob(const std::string &blob_name) const
{
  return _blobNameToOpOutput.at(blob_name);
}

void CaffeImporter::setOutputForBlob(const std::string &blob_name, mir::Operation::Output *output)
{
  const auto it = _blobNameToOpOutput.find(blob_name);
  if (it != _blobNameToOpOutput.cend())
  {
    // caffe input blob name could be same as output blob name, and next line will overwrite
    // '_blobNameToOpOutput' element, but in all networks that I saw it was not a problem
    it->second->setName("");
  }

  // Do not overwrite the name in case of fall-through layers (ex. Dropout, Split).
  // TODO Find a way to handle it properly.
  if (output->getName().empty())
    output->setName(blob_name);

  _blobNameToOpOutput[blob_name] = output;
}

void CaffeImporter::setGraphOutputs(mir::Graph *graph)
{
  // TODO For now, we assume that:
  //      - there is exactly one output;
  //      - the output is from the last layer.
  const auto &last_layer = *_net->layer().rbegin();
  auto output = getOutputForBlob(last_layer.top(0));
  graph->create<mir::ops::OutputOp>(output);
}

const std::map<std::string, CaffeOpType> CaffeImporter::_operatorTypes = {
  {"AbsVal", CaffeOpType::absVal},
  {"Accuracy", CaffeOpType::accuracy},
  {"ArgMax", CaffeOpType::argMax},
  {"BatchNorm", CaffeOpType::batchNorm},
  {"BatchReindex", CaffeOpType::batchReindex},
  {"Bias", CaffeOpType::bias},
  {"BNLL", CaffeOpType::BNLL},
  {"Clip", CaffeOpType::clip},
  {"Concat", CaffeOpType::concat},
  {"ContrastiveLoss", CaffeOpType::contrastiveLoss},
  {"Convolution", CaffeOpType::convolution},
  {"Crop", CaffeOpType::crop},
  {"Data", CaffeOpType::data},
  {"Deconvolution", CaffeOpType::deconvolution},
  {"Dropout", CaffeOpType::dropout},
  {"DummyData", CaffeOpType::dummyData},
  {"Eltwise", CaffeOpType::eltwise},
  {"ELU", CaffeOpType::ELU},
  {"Embed", CaffeOpType::embed},
  {"EuclidianLoss", CaffeOpType::euclidianLoss},
  {"Exp", CaffeOpType::exp},
  {"Filter", CaffeOpType::filter},
  {"Flatten", CaffeOpType::flatten},
  {"HDF5Data", CaffeOpType::HDF5Data},
  {"HDF5Output", CaffeOpType::HDF5Output},
  {"HingeLoss", CaffeOpType::hingeLoss},
  {"Im2Col", CaffeOpType::im2Col},
  {"ImageData", CaffeOpType::imageData},
  {"InfogainLoss", CaffeOpType::infogainLoss},
  {"InnerProduct", CaffeOpType::innerProduct},
  {"Input", CaffeOpType::input},
  {"Log", CaffeOpType::log},
  {"LRN", CaffeOpType::LRN},
  {"LSTM", CaffeOpType::LSTM},
  {"MemoryData", CaffeOpType::memoryData},
  {"MultinomialLogisticLoss", CaffeOpType::multinomialLogisticLoss},
  {"MVN", CaffeOpType::MVN},
  {"Parameter", CaffeOpType::parameter},
  {"Pooling", CaffeOpType::pooling},
  {"Power", CaffeOpType::power},
  {"PReLU", CaffeOpType::PReLU},
  {"Python", CaffeOpType::python},
  {"Recurrent", CaffeOpType::recurrent},
  {"Reduction", CaffeOpType::reduction},
  {"ReLU", CaffeOpType::ReLU},
  {"Reshape", CaffeOpType::reshape},
  {"RNN", CaffeOpType::RNN},
  {"Scale", CaffeOpType::scale},
  {"SigmoidCrossEntropyLoss", CaffeOpType::sigmoidCrossEntropyLoss},
  {"Sigmoid", CaffeOpType::sigmoid},
  {"Silence", CaffeOpType::silence},
  {"Softmax", CaffeOpType::softmax},
  {"SoftmaxWithLoss", CaffeOpType::softmaxWithLoss},
  {"SPP", CaffeOpType::SPP},
  {"Split", CaffeOpType::split},
  {"Slice", CaffeOpType::slice},
  {"TanH", CaffeOpType::tanh},
  {"Threshold", CaffeOpType::threshold},
  {"Tile", CaffeOpType::tile},
  {"WindowData", CaffeOpType::windowData}};
} // namespace

std::unique_ptr<mir::Graph> importModelFromBinaryFile(const std::string &filename)
{
  CaffeImporter importer;
  return importer.importModelFromBinaryFile(filename);
}

std::unique_ptr<mir::Graph> importModelFromTextFile(const std::string &filename)
{
  CaffeImporter importer;
  return importer.importModelFromTextFile(filename);
}

std::unique_ptr<mir::Graph> loadModel(const std::string &filename)
{
  return importModelFromBinaryFile(filename);
}

} // namespace mir_caffe
