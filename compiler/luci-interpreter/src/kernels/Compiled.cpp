/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Compiled.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

template <typename TensorT>
static void add_arguemnt_descriptions(std::vector<int> &ranks, std::vector<std::vector<int>> &dims, const std::vector<TensorT *> &tensors)
{
  for (int i = 0; i < tensors.size(); ++i)
  {
    int rank = tensors[i]->shape().num_dims();
    ranks.push_back(rank);
    dims.emplace_back(rank);
    for (int j = 0; j < rank; ++j)
    {
      dims.back()[j] = tensors[i]->shape().dim(j);
    }
  }
}

Compiled::Compiled(std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs, const CompiledParams &params)
  : KernelWithParams<CompiledParams>(std::move(inputs), std::move(outputs), params)
{
  std::vector<int> ranks;
  std::vector<std::vector<int>> dims;
  add_arguemnt_descriptions(ranks, dims, inputs);
  add_arguemnt_descriptions(ranks, dims, outputs);
  std::vector<int*> raw_dims;
  for (int i = 0; i < dims.size(); ++i)
  {
    raw_dims.push_back(dims[i].data());
  }
  _impl = params.constructor(ranks.data(), raw_dims.data());
}

Compiled::~Compiled()
{
  _params.destructor(&_impl);
}

void Compiled::configure()
{
  for (int i = 0; i < _params.output_shapes.size(); ++i)
  {
    output(i)->resize(_params.output_shapes[i]);
  }
}

void Compiled::execute() const
{
  std::vector<void *> args(num_inputs() + num_outputs());
  for (int i = 0; i < num_inputs(); ++i)
  {
    args[i] = const_cast<void *>(input(i)->data<void>());
  }
  for (int i = 0; i < num_outputs(); ++i)
  {
    args[num_inputs() + i] = output(i)->data<void>();
  }
  _impl.wrapper(_impl.configuration, args.data());
}

} // namespace kernels
} // namespace luci_interpreter
