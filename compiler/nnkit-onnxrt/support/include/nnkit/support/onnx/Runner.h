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

#ifndef __NNKIT_SUPPORT_ONNX_RUNNER_H__
#define __NNKIT_SUPPORT_ONNX_RUNNER_H__

#include "nnkit/support/onnx/Allocator.h"
#include "nnkit/support/onnx/TensorSet.h"

#include <onnxruntime_c_api.h>

#include <memory>

namespace nnkit
{
namespace support
{
namespace onnx
{

class Runner
{
public:
  Runner(const std::string &path);
  ~Runner(void);

  void prepareInputs(void);
  void prepareOutputs(void);

  TensorSet &inputs(void) { return *_inputs; }
  TensorSet &outputs(void) { return *_outputs; }

  void run(void);

public:
  // Disallow copy
  Runner(const Runner &) = delete;
  Runner &operator=(const Runner &) = delete;

private:
  OrtEnv *_env;
  OrtSession *_session;

  std::unique_ptr<Allocator> _allocator;

  std::unique_ptr<TensorSet> _inputs;
  std::unique_ptr<TensorSet> _outputs;
};

} // namespace onnx
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_ONNX_RUNNER_H__
