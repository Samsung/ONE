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

#include "nnkit/support/moco/tf/Backend.h"

#include "InputTensorContext.h"
#include "OutputTensorContext.h"

#include "nnkit/TensorContext.h"
#include "nnkit/support/tftestinfo/ParsedTensor.h"
#include "nnkit/support/tftestinfo/TensorInfoParser.h"

#include <moco/tf/Frontend.h>
#include <moco/Names.h>
#include <stdex/Memory.h>

#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <utility> // std::move
#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace moco
{
namespace tf
{

void Backend::setInputOutputFromGraph(const std::unique_ptr<loco::Graph> &loco_graph,
                                      ParsedTensors &parsed_tensors)
{
  auto inputs = loco_graph.get()->inputs();
  auto outputs = loco_graph.get()->outputs();
  uint32_t input_idx = 0;
  uint32_t output_idx = 0;
  for (auto &parsed_tensor : parsed_tensors)
  {
    if (parsed_tensor->kind() == ParsedTensor::Kind::Input)
    {
      if (!parsed_tensor->hasShape())
      {
        auto input_shape = inputs->at(input_idx++)->shape();

        uint32_t size = input_shape->rank();
        parsed_tensor->mutable_shape().resize(size);
        for (uint32_t d = 0; d < size; d++)
        {
          parsed_tensor->mutable_shape().dim(d) = input_shape->dim(d).value();
        }
      }
      _inputs.emplace_back(std::move(parsed_tensor));
    }
    else // Output
    {
      if (!parsed_tensor->hasShape())
      {
        auto output_shape = outputs->at(output_idx++)->shape();

        uint32_t size = output_shape->rank();
        parsed_tensor->mutable_shape().resize(size);
        for (uint32_t d = 0; d < size; d++)
        {
          parsed_tensor->mutable_shape().dim(d) = output_shape->dim(d).value();
        }
      }
      _outputs.emplace_back(std::move(parsed_tensor));
    }
  }
}

Backend::Backend(const char *pb_path, const char *info_path)
{
  // read test.info
  ::moco::ModelSignature sig;

  auto parsed_tensors = nnkit::support::tftestinfo::parse(info_path);

  for (auto &parsed_tensor : parsed_tensors)
  {
    if (parsed_tensor->kind() == ParsedTensor::Kind::Input)
    {
      sig.add_input(::moco::TensorName(parsed_tensor->name()));
    }
    else
    {
      sig.add_output(::moco::TensorName(parsed_tensor->name()));
    }
    if (parsed_tensor->hasShape())
      sig.shape(parsed_tensor->name(), parsed_tensor->shape());
  }

  // get loco::Graph
  ::moco::tf::Frontend moco;

  // After converting, all shapes will be determined.
  auto loco_graph = moco.load(sig, pb_path, ::moco::tf::Frontend::FileType::Binary);

  // Set input and output from loco graph.
  setInputOutputFromGraph(loco_graph, parsed_tensors);

  // set member vars
  _loco_graph = std::move(loco_graph);
  _sess = stdex::make_unique<locomotiv::Session>(_loco_graph.get());
}

void Backend::prepare(const std::function<void(nnkit::TensorContext &)> &f)
{
  using nncc::core::ADT::tensor::Buffer;
  using nncc::core::ADT::tensor::LexicalLayout;
  using nncc::core::ADT::tensor::make_buffer;

  // allocate memory for inputs of loco interpreter
  std::vector<std::unique_ptr<Buffer<float>>> buf_list; // TODO Support more types other than float

  for (int n = 0; n < _inputs.size(); n++)
  {
    auto buf = make_buffer<float, LexicalLayout>(_inputs.at(n)->shape());
    buf_list.emplace_back(stdex::make_unique<nncc::core::ADT::tensor::Buffer<float>>(buf));
  }

  // fill test input values
  InputTensorContext ctx(_inputs, buf_list);
  f(ctx);

  // set input of locomotiv
  for (int n = 0; n < buf_list.size(); n++)
  {
    auto buf = buf_list.at(n).get();
    auto node_data = locomotiv::make_data(*buf);
    _sess->set_input(n, std::move(node_data));
  }
}

void Backend::run(void) { _sess->infer(); }

void Backend::teardown(const std::function<void(nnkit::TensorContext &)> &f)
{
  // get output
  OutputTensorContext ctx(_outputs, _sess.get());
  f(ctx);
}

} // namespace tf
} // namespace moco
} // namespace support
} // namespace nnkit
