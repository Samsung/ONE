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

#include "PackCommand.hpp"
#include "Support.hpp"

#include <tensorflow/core/framework/graph.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace
{

template <typename T> void pack(tensorflow::TensorProto *);

template <> void pack<float>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->float_val().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->float_val().size();
  }

  if (input_tensor->float_val().size() == 0)
  {
    // There may be tensor_content and we don't need to do anything as it is
    // already packed format
  }
  else if (input_tensor->float_val().size() == input_flat_size)
  {
    input_tensor->clear_tensor_content();

    std::vector<float> tensor_content;
    for (int i = 0; i < input_flat_size; ++i)
    {
      tensor_content.push_back(input_tensor->float_val(i));
    }

    input_tensor->set_tensor_content(std::string(
      reinterpret_cast<const char *>(tensor_content.data()), sizeof(float) * input_flat_size));

    input_tensor->clear_float_val();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in pack<float>."};
    // TODO: support for these
  }
}

template <> void pack<int32_t>(tensorflow::TensorProto *input_tensor)
{
  const auto &input_shape = input_tensor->tensor_shape();
  assert(input_shape.dim_size() <= 6);
  int input_flat_size = tfkit::tf::GetElementCount(input_shape);

  // Adjust where shape is not set but actual value exist
  if (input_tensor->int_val().size() > 0 && input_flat_size == -1)
  {
    input_flat_size = input_tensor->int_val().size();
  }

  if (input_tensor->int_val().size() == 0)
  {
    // There may be tensor_content and we don't need to do anything as it is
    // already packed format
  }
  else if (input_tensor->int_val().size() == input_flat_size)
  {
    input_tensor->clear_tensor_content();

    std::vector<int32_t> tensor_content;
    for (int i = 0; i < input_flat_size; ++i)
    {
      tensor_content.push_back(input_tensor->int_val(i));
    }

    input_tensor->set_tensor_content(std::string(
      reinterpret_cast<const char *>(tensor_content.data()), sizeof(int32_t) * input_flat_size));

    input_tensor->clear_int_val();
  }
  else
  {
    throw std::runtime_error{"Number of elements mismatch in pack<int32_t>."};
    // TODO: support for these
  }
}

void pack(tensorflow::GraphDef &graph_def)
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
          pack<float>(tensor);
          break;
        case tensorflow::DT_INT32:
          pack<int32_t>(tensor);
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

int PackCommand::run(int argc, const char *const *argv) const
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

  // convert float_val to tensor_content
  pack(graph_def);

  google::protobuf::io::OstreamOutputStream os{ioconfig->out()};
  google::protobuf::TextFormat::Print(graph_def, &os);

  return 0;
}

} // namespace tfkit
