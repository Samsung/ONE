/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "../Registration.h"

namespace onert
{
namespace interp
{
namespace
{

void prepare(ExecEnv *env, const ir::Operation &node)
{
  const auto in_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  // Unspecified shape is not supported in operation node spec now
  const auto output_info = env->graph().operands().at(out_index).info();
  env->allocateAndShareIfNeeded(out_index, output_info, in_index);

  assert(output_info.total_size() == env->graph().operands().at(in_index).info().total_size());
}

void invoke(const ExecEnv *env, const ir::Operation &node)
{
  const auto in_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  if (env->tensorAt(in_index)->bufferRO() == env->tensorAt(out_index)->bufferRO())
  {
    // Same data
    return;
  }

  const auto output_info = env->graph().operands().at(out_index).info();
  memcpy(env->tensorAt(out_index)->buffer(), env->tensorAt(in_index)->bufferRO(),
         output_info.total_size());
}

} // namespace

OpKernel *getReshape()
{
  static OpKernel kernel = {prepare, invoke};
  return &kernel;
}

} // namespace interp
} // namespace onert
