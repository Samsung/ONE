/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_KERNEL_FULLYCONNECTEDLAYER_H__
#define __ONERT_BACKEND_CPU_KERNEL_FULLYCONNECTEDLAYER_H__

#include "../operand/Tensor.h"
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace nnfw
{
namespace cker
{
class FCTempArena;
}
} // namespace nnfw

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class FullyConnectedLayer : public ::onert::exec::IFunction
{
public:
  FullyConnectedLayer();
  ~FullyConnectedLayer();

public:
  void fullyConnectedFloat32();

  void fullyConnectedQuant8();

  void fullyConnectedHybrid();

  void configure(const ITensor *input, const ITensor *weights, const ITensor *bias,
                 ir::Activation activation, ITensor *output);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  const ITensor *_input;
  const ITensor *_weights;
  const ITensor *_bias;
  ITensor *_output;

  ir::Activation _activation;
  std::unique_ptr<nnfw::cker::FCTempArena> _temp_arena;
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_FULLYCONNECTEDLAYER_H__
