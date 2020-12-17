/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

// Code here refers https://github.com/Neargye/hello_tf_c_api
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2019 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "nnkit/support/tf/Runner.h"

#include "nnkit/support/tftestinfo/ParsedTensor.h"
#include "nncc/core/ADT/tensor/Shape.h"

#include <tensorflow/c/c_api.h>

#include <vector>
#include <cassert>
#include <cstring> // std::memcpy()
#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace tf
{

using nncc::core::ADT::tensor::num_elements;
using nnkit::support::tftestinfo::ParsedTensor;

namespace
{
TF_Tensor *create_tensor(const TF_DataType data_type, const std::int64_t *dims,
                         const std::size_t num_dims, const void *data, const std::size_t len)
{
  if (dims == nullptr || data == nullptr)
    return nullptr;

  TF_Tensor *tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
  if (tensor == nullptr)
    return nullptr;

  void *tensor_data = TF_TensorData(tensor);
  if (tensor_data == nullptr)
  {
    TF_DeleteTensor(tensor);
    return nullptr;
  }

  std::memcpy(tensor_data, data, std::min(len, TF_TensorByteSize(tensor)));

  return tensor;
}

void deallocate_buffer(void *data, size_t)
{
  assert(data);
  std::free(data);
}

TF_Buffer *build_TFBuffer(const char *file)
{
  const auto f = std::fopen(file, "rb");

  if (f == nullptr)
    throw std::runtime_error(std::string("cannot open ") + file);

  std::fseek(f, 0, SEEK_END); // to get file size
  const auto fsize = ftell(f);

  std::fseek(f, 0, SEEK_SET);

  if (fsize < 1)
  {
    std::fclose(f);
    throw std::runtime_error(std::string("file read error:  ") + file);
  }

  const auto data = std::malloc(fsize);
  std::fread(data, fsize, 1, f);
  std::fclose(f);

  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = deallocate_buffer;

  return buf;
}

} // namespace

Runner::Runner(const char *pb_path)
{
  // initialize member variables
  _sess = nullptr;
  _graph = TF_NewGraph();
  _status = TF_NewStatus();

  // import graph from file
  TF_Buffer *buffer = build_TFBuffer(pb_path);
  if (buffer == nullptr)
    throw std::runtime_error("Can't read buffer from file");

  TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(_graph, buffer, opts, _status);

  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(_status) != TF_OK) // TODO Consider wrapper to prevent memory leak
    throw std::runtime_error("Can't import GraphDef");
}

Runner::~Runner()
{
  if (_graph)
    TF_DeleteGraph(_graph);

  if (_sess)
  {
    TF_CloseSession(_sess, _status);
    TF_DeleteSession(_sess, _status);
  }

  for (auto tensor : _input_tensors)
    TF_DeleteTensor(tensor);

  for (auto tensor : _output_tensors)
    TF_DeleteTensor(tensor);

  TF_DeleteStatus(_status);
}

bool Runner::getTensorShapeFromGraphDef(const std::unique_ptr<ParsedTensor> &tensor,
                                        angkor::TensorShape &shape)
{
  assert(!tensor->hasShape());
  TF_Output tensor_op = {TF_GraphOperationByName(_graph, tensor->nodeName().c_str()),
                         tensor->tensorIndex()};

  if (tensor_op.oper == nullptr)
    return false;

  int dim_size = TF_GraphGetTensorNumDims(_graph, tensor_op, _status);
  if (dim_size == -1)
    return false;
  int64_t dims[dim_size];

  TF_GraphGetTensorShape(_graph, tensor_op, dims, dim_size, _status);

  shape.resize(dim_size);
  for (int d = 0; d < dim_size; d++)
  {
    if (dims[d] == -1)
      return false;
    shape.dim(d) = dims[d];
  }
  return true;
}

bool Runner::getTensorDtypeFromGraphDef(const std::unique_ptr<ParsedTensor> &tensor,
                                        Runner::DataType &dtype)
{
  TF_Output tensor_op = {TF_GraphOperationByName(_graph, tensor->nodeName().c_str()),
                         tensor->tensorIndex()};

  if (tensor_op.oper == nullptr)
    return false;

  TF_DataType tf_dtype = TF_OperationOutputType(tensor_op);

  switch (tf_dtype)
  {
    case TF_DataType::TF_FLOAT:
      dtype = DataType::FLOAT;
      break;
    case TF_DataType::TF_UINT8:
      dtype = DataType::U8;
      break;
    case TF_DataType::TF_UINT16:
      dtype = DataType::U16;
      break;
    case TF_DataType::TF_UINT32:
      dtype = DataType::U32;
      break;
    case TF_DataType::TF_UINT64:
      dtype = DataType::U64;
      break;
    case TF_DataType::TF_INT8:
      dtype = DataType::S8;
      break;
    case TF_DataType::TF_INT16:
      dtype = DataType::S16;
      break;
    case TF_DataType::TF_INT32:
      dtype = DataType::S32;
      break;
    case TF_DataType::TF_INT64:
      dtype = DataType::S64;
      break;
    default:
      dtype = DataType::Unknown;
      return false;
  }
  return true;
}

void Runner::prepareInputs(const std::vector<std::unique_ptr<ParsedTensor>> &inputs,
                           TensorDataMap &data_map)
{
  assert(_graph);

  for (const auto &tensor : inputs)
  {
    TF_Output input_op = {TF_GraphOperationByName(_graph, tensor->nodeName().c_str()),
                          tensor->tensorIndex()};

    if (input_op.oper == nullptr)
      throw std::runtime_error("Can't init input_op : " + tensor->name());

    std::vector<int64_t> shape;
    for (int r = 0; r < tensor->shape().rank(); r++)
      shape.emplace_back(tensor->shape().dim(r));

    int size = 0;
    if (tensor->isFloatTensor())
      size = sizeof(float);
    else
      throw std::runtime_error("Not supported tensor type");

    TF_Tensor *input_tensor =
      create_tensor(TF_FLOAT, shape.data(), shape.size(), data_map.data(tensor.get()),
                    num_elements(tensor->shape()) * size);

    _input_ops.emplace_back(input_op);
    _input_tensors.emplace_back(input_tensor);
  }
}

void Runner::prepareOutputs(const std::vector<std::unique_ptr<ParsedTensor>> &outputs)
{
  assert(_graph);

  for (const auto &tensor : outputs)
  {
    TF_Output output_op = {TF_GraphOperationByName(_graph, tensor->nodeName().c_str()),
                           tensor->tensorIndex()};

    if (output_op.oper == nullptr)
      throw std::runtime_error("Can't init output_op : " + tensor->name());

    _output_ops.emplace_back(output_op);
  }

  _output_tensors.resize(_output_ops.size());
}

void Runner::run()
{
  assert(_graph);
  assert(_output_ops.size() > 0);

  TF_SessionOptions *options = TF_NewSessionOptions();
  _sess = TF_NewSession(_graph, options, _status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(_status) != TF_OK)
    throw std::runtime_error(TF_Message(_status));

  TF_SessionRun(_sess,
                nullptr, // Run options.
                _input_ops.data(), _input_tensors.data(), _input_ops.size(), _output_ops.data(),
                _output_tensors.data(), _output_ops.size(), nullptr,
                0,       // Target operations, number of targets.
                nullptr, // Run metadata.
                _status  // Output status.
  );

  if (TF_GetCode(_status) != TF_OK)
    throw std::runtime_error(TF_Message(_status));

  TF_CloseSession(_sess, _status);
  TF_DeleteSession(_sess, _status);
  _sess = nullptr;
}

} // namespace tf
} // namespace support
} // namespace nnkit
