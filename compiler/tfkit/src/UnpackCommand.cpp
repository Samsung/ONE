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

#include "UnpackCommand.hpp"
#include "Support.hpp"

#include <tensorflow/core/framework/graph.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cassert>
#include <stdexcept>

namespace
{

template <typename T> void unpack(tensorflow::TensorProto *);

template <> void unpack<float>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->tensor_content().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->tensor_content().size() / sizeof(float);
  }

  if (input_tensor->tensor_content().size() == 0)
  {
    // Do nothing as there is no tensor content to unpack
  }
  else if (input_tensor->tensor_content().size() == input_flat_size * sizeof(float))
  {
    input_tensor->clear_float_val();

    const float *tensor_content =
      reinterpret_cast<const float *>(input_tensor->tensor_content().data());
    for (int i = 0; i < input_flat_size; i++)
    {
      input_tensor->add_float_val(tensor_content[i]);
    }
    input_tensor->clear_tensor_content();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in unpack<float>."};
    // TODO: support for these
  }
}

template <> void unpack<int32_t>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->tensor_content().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->tensor_content().size() / sizeof(int32_t);
  }

  if (input_tensor->tensor_content().size() == 0)
  {
    // Do nothing as there is no tensor content to unpack
  }
  else if (input_tensor->tensor_content().size() == input_flat_size * sizeof(int32_t))
  {
    input_tensor->clear_int_val();

    const int32_t *tensor_content =
      reinterpret_cast<const int32_t *>(input_tensor->tensor_content().data());
    for (int i = 0; i < input_flat_size; i++)
    {
      input_tensor->add_int_val(tensor_content[i]);
    }
    input_tensor->clear_tensor_content();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in unpack<int32_t>."};
    // TODO: support for these
  }
}

template <> void unpack<int8_t>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->tensor_content().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->tensor_content().size() / sizeof(int8_t);
  }

  if (input_tensor->tensor_content().size() == 0)
  {
    // Do nothing as there is no tensor content to unpack
  }
  else if (input_tensor->tensor_content().size() == input_flat_size * sizeof(int8_t))
  {
    input_tensor->clear_int_val();

    const int8_t *tensor_content =
      reinterpret_cast<const int8_t *>(input_tensor->tensor_content().data());
    for (int i = 0; i < input_flat_size; i++)
    {
      input_tensor->add_int_val(tensor_content[i]);
    }
    input_tensor->clear_tensor_content();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in unpack<int8_t>."};
    // TODO: support for these
  }
}

template <> void unpack<bool>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->tensor_content().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->tensor_content().size() / sizeof(bool);
  }

  if (input_tensor->tensor_content().size() == 0)
  {
    // Do nothing as there is no tensor content to unpack
  }
  else if (input_tensor->tensor_content().size() == input_flat_size * sizeof(bool))
  {
    input_tensor->clear_bool_val();

    const bool *tensor_content =
      reinterpret_cast<const bool *>(input_tensor->tensor_content().data());
    for (int i = 0; i < input_flat_size; i++)
    {
      input_tensor->add_bool_val(tensor_content[i]);
    }
    input_tensor->clear_tensor_content();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in unpack<bool>."};
    // TODO: support for these
  }
}

void unpack(tensorflow::GraphDef &graph_def)
{
  auto nodes = graph_def.mutable_node();
  for (int i = 0; i < nodes->size(); ++i)
  {
    tensorflow::NodeDef *n = nodes->Mutable(i);
    // TODO: handle for other operators
    if (n->op() == "Const")
    {
      const auto dtype = tfkit::tf::GetDataTypeAttr(*n, "dtype");
      tensorflow::TensorProto *tensor = tfkit::tf::GetTensorAttr(*n, "value");

      switch (dtype)
      {
        case tensorflow::DT_FLOAT:
          unpack<float>(tensor);
          break;
        case tensorflow::DT_INT32:
          unpack<int32_t>(tensor);
          break;
        case tensorflow::DT_INT8:
          unpack<int8_t>(tensor);
          break;
        case tensorflow::DT_BOOL:
          unpack<bool>(tensor);
          break;
        default:
          throw std::runtime_error{"Unsupported dtype"};
      }
    }
  }
}

} // namespace

namespace tfkit
{

int UnpackCommand::run(int argc, const char *const *argv) const
{
  tensorflow::GraphDef graph_def;

  CmdArguments cmdargs(argc, argv);

  auto ioconfig = make_ioconfig(cmdargs);

  google::protobuf::io::IstreamInputStream is{ioconfig->in()};

  if (!google::protobuf::TextFormat::Parse(&is, &graph_def))
  {
    std::cerr << "ERROR: Failed to parse prototxt" << std::endl;
    return 255;
  }

  // convert tensor_content to float_val
  unpack(graph_def);

  google::protobuf::io::OstreamOutputStream os{ioconfig->out()};
  google::protobuf::TextFormat::Print(graph_def, &os);

  return 0;
}

} // namespace tfkit
