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

#include "ONNXImporterImpl.h"
#include "ONNXHelpers.h"
#include "ONNXOpRegistration.h"
#include "onnx/onnx.pb.h"

#include "mir/Shape.h"
#include "mir/TensorUtil.h"

#include "mir/ops/ConstantOp.h"

#include <fcntl.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

namespace mir_onnx
{

namespace
{

class ONNXImporterImpl final
{
public:
  ONNXImporterImpl();
  ~ONNXImporterImpl();
  /// @brief Load the model and convert it into a MIR Graph.
  std::unique_ptr<mir::Graph> importModelFromBinaryFile(const std::string &filename);
  std::unique_ptr<mir::Graph> importModelFromTextFile(const std::string &filename);

private:
  std::unique_ptr<mir::Graph> createIR();
  void createGraphInputs();
  void collectUnsupportedOps();
  std::unique_ptr<onnx::ModelProto> _model;
  std::unique_ptr<ConverterContext> _converterCtx;
  std::unique_ptr<ModelContext> _modelCtx;
  std::unique_ptr<mir::Graph> _graph;
};

ONNXImporterImpl::ONNXImporterImpl() { registerSupportedOps(); }

ONNXImporterImpl::~ONNXImporterImpl() = default;

void loadModelFromBinaryFile(const std::string &filename, onnx::ModelProto *model)
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

  if (!model->ParseFromCodedStream(&coded_stream))
    throw std::runtime_error("Couldn't parse file \"" + filename + "\".");

  // If the file has not been consumed entirely, assume that the file is in the wrong format.
  if (!coded_stream.ConsumedEntireMessage())
    throw std::runtime_error("File \"" + filename + "\" has not been consumed entirely.");
}

void loadModelFromTextFile(const std::string &filename, onnx::ModelProto *model)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int file_handle = open(filename.c_str(), O_RDONLY);

  if (file_handle == -1)
    throw std::runtime_error("Couldn't open file \"" + filename + "\": " + std::strerror(errno) +
                             ".");

  google::protobuf::io::FileInputStream file_stream(file_handle);
  file_stream.SetCloseOnDelete(true);

  if (!google::protobuf::TextFormat::Parse(&file_stream, model))
    throw std::runtime_error("Couldn't parse file \"" + filename + "\".");
}

std::unique_ptr<mir::Graph> ONNXImporterImpl::importModelFromBinaryFile(const std::string &filename)
{
  _model = std::make_unique<onnx::ModelProto>();
  loadModelFromBinaryFile(filename, _model.get());
  _modelCtx = std::make_unique<ModelContext>(_model.get());
  collectUnsupportedOps();
  return createIR();
}

std::unique_ptr<mir::Graph> ONNXImporterImpl::importModelFromTextFile(const std::string &filename)
{
  _model = std::make_unique<onnx::ModelProto>();
  loadModelFromTextFile(filename, _model.get());
  _modelCtx = std::make_unique<ModelContext>(_model.get());
  collectUnsupportedOps();
  return createIR();
}

void ONNXImporterImpl::collectUnsupportedOps()
{
  std::set<std::pair<std::string, int64_t>> problems_op_set;

  for (int i = 0; i < _model->graph().node_size(); i++)
  {
    const auto &onnx_node = _model->graph().node(i);
    assert(onnx_node.has_op_type());
    const auto &op_type = onnx_node.op_type();
    auto opset = _modelCtx->getDomainOpsetVersion(onnx_node.domain());

    NodeConverterRegistry::ConverterFunc converter =
      NodeConverterRegistry::getInstance().lookup(op_type, opset);

    if (converter == nullptr)
      problems_op_set.emplace(op_type, opset);
  }
  if (!problems_op_set.empty())
  {
    std::cerr << "The following operators are not supported:\n";
    for (const auto &op : problems_op_set)
      std::cerr << op.first << " opset " << op.second << std::endl;
    throw std::runtime_error("Unsupported operators found");
  }
}

void ONNXImporterImpl::createGraphInputs()
{
  const auto &graph = _model->graph();
  const auto &initializer = graph.initializer();

  // Create all initializer Tensors
  for (const auto &tensor : initializer)
  {
    const auto mir_tensor = createTensor(&tensor);
    auto *op = _graph->create<mir::ops::ConstantOp>(mir_tensor);
    _converterCtx->setOutput(tensor.name(), op->getOutput(0));
  }

  for (const auto &input : graph.input())
  {
    assert(input.has_name());

    if (_converterCtx->getOutput(input.name()) == nullptr)
    {
      const auto &onnx_input_shape = input.type().tensor_type().shape();
      mir::Shape shape(onnx_input_shape.dim_size());
      for (int i = 0; i < onnx_input_shape.dim_size(); i++)
      {
        assert(onnx_input_shape.dim(i).has_dim_value());
        shape.dim(i) = static_cast<int32_t>(onnx_input_shape.dim(i).dim_value());
      }

      auto elem_type = onnxDataTypeToMirDataType(
        (onnx::TensorProto_DataType)input.type().tensor_type().elem_type());
      mir::TensorType type{elem_type, shape};
      auto *op = _graph->create<mir::ops::InputOp>(type);
      _converterCtx->setOutput(input.name(), op->getOutput(0));
    }
  }
}

std::unique_ptr<mir::Graph> ONNXImporterImpl::createIR()
{
  _graph = std::make_unique<mir::Graph>();
  _converterCtx = std::make_unique<ConverterContext>(_graph.get());

  createGraphInputs();

  // Forming partially ordered computation graph
  for (const auto &onnx_node : _model->graph().node())
  {
    assert(onnx_node.has_op_type());
    auto &op_type = onnx_node.op_type();
    auto opset = _modelCtx->getDomainOpsetVersion(onnx_node.domain());
    // Get converter
    NodeConverterRegistry::ConverterFunc converter =
      NodeConverterRegistry::getInstance().lookup(op_type, opset);
    assert(converter != nullptr);
    converter(onnx_node, _converterCtx.get());
  }
  // Set graph outputs
  const auto &outputs = _model->graph().output();
  for (const auto &output : outputs)
  {
    assert(output.has_name());
    auto mir_output = _converterCtx->getOutput(output.name());
    if (mir_output == nullptr)
      throw std::runtime_error("Bad output name!");

    _graph->create<mir::ops::OutputOp>(mir_output);
  }

  return std::move(_graph);
}

} // namespace

std::unique_ptr<mir::Graph> importModelFromBinaryFile(const std::string &filename)
{
  ONNXImporterImpl importer;
  return importer.importModelFromBinaryFile(filename);
}

std::unique_ptr<mir::Graph> importModelFromTextFile(const std::string &filename)
{
  ONNXImporterImpl importer;
  return importer.importModelFromTextFile(filename);
}

std::unique_ptr<mir::Graph> loadModel(const std::string &filename)
{
  return importModelFromBinaryFile(filename);
}

} // namespace mir_onnx
