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

#include "nnkit/support/tf/Backend.h"

#include "nnkit/support/tftestinfo/ParsedTensor.h"
#include "nnkit/support/tftestinfo/TensorInfoParser.h"
#include "nnkit/support/tf/TensorDataMap.h"
#include "nnkit/support/tf/TensorContext.h"
#include "nnkit/support/tf/Runner.h"

#include <angkor/TensorShape.h>

#include <nnkit/Backend.h>

#include <cstring> // memcpy

namespace nnkit
{
namespace support
{
namespace tf
{

using nnkit::support::tftestinfo::ParsedTensor;

Backend::Backend(const char *pb_path, const char *info_path) : _tf_runner(pb_path)
{
  auto parsed_tensors = nnkit::support::tftestinfo::parse(info_path);
  for (auto &parsed_tensor : parsed_tensors)
  {
    if (parsed_tensor->kind() == ParsedTensor::Kind::Input)
    {
      // user didn't specify input
      if (!parsed_tensor->hasShape())
      {
        angkor::TensorShape shape;
        if (!_tf_runner.getTensorShapeFromGraphDef(parsed_tensor, shape))
          throw oops::UserExn(
            "Info you provided may be wrong or not enough. Please check the info file.");

        parsed_tensor->mutable_shape().resize(shape.rank());
        for (int r = 0; r < shape.rank(); r++)
        {
          parsed_tensor->mutable_shape().dim(r) = shape.dim(r);
        }
      }
      _inputs.emplace_back(std::move(parsed_tensor));
    }
    else
      _outputs.emplace_back(std::move(parsed_tensor));
  }
}

void Backend::prepare(const std::function<void(nnkit::TensorContext &)> &f)
{
  for (const auto &input_tensor : _inputs)
    _data_map.allocate(input_tensor.get());

  TensorContext ctx(_inputs, _data_map);
  f(ctx); // fill values

  _tf_runner.prepareInputs(_inputs, _data_map);
  _tf_runner.prepareOutputs(_outputs);
}

void Backend::run(void)
{
  _tf_runner.run();

  // get result
  auto actual_outputs = _tf_runner.output();

  for (int n = 0; n < _outputs.size(); n++)
  {
    auto actual = actual_outputs[n];
    const size_t byte_size = TF_TensorByteSize(actual);
    const uint8_t *tf_data = reinterpret_cast<const uint8_t *>(TF_TensorData(actual));

    const uint32_t shape_rank = TF_NumDims(actual);
    _outputs[n]->mutable_shape().resize(shape_rank);
    for (uint32_t r = 0; r < shape_rank; r++)
    {
      _outputs[n]->mutable_shape().dim(r) = TF_Dim(actual, r);
    }
    uint8_t *dest = _data_map.allocate(_outputs[n].get());

    std::memcpy(dest, tf_data, byte_size);
  }
}

void Backend::teardown(const std::function<void(nnkit::TensorContext &)> &f)
{
  TensorContext ctx(_outputs, _data_map);
  f(ctx);
}

} // namespace tf
} // namespace support
} // namespace nnkit
