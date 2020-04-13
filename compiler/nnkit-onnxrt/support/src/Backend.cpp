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

#include "nnkit/support/onnx/Backend.h"
#include "nnkit/support/onnx/TensorContext.h"

namespace nnkit
{
namespace support
{
namespace onnx
{

void Backend::prepare(const std::function<void(nnkit::TensorContext &)> &f)
{
  // Prepare input and output tensors
  _runner.prepareInputs();
  _runner.prepareOutputs();

  TensorContext ctx(_runner.inputs());
  f(ctx);
}

void Backend::run(void) { _runner.run(); }

void Backend::teardown(const std::function<void(nnkit::TensorContext &)> &f)
{
  TensorContext ctx(_runner.outputs());
  f(ctx);
}

} // namespace onnx
} // namespace support
} // namespace nnkit
