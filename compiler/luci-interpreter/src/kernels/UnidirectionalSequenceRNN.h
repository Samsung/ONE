/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCERNN_H
#define LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCERNN_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class UnidirectionalSequenceRNN : public KernelWithParams<UnidirectionalSequenceRNNParams>
{
public:
  UnidirectionalSequenceRNN(const Tensor *input, const Tensor *input_weights,
                            const Tensor *input_recurrent_weights, const Tensor *input_bias,
                            const Tensor *input_hidden_state, Tensor *output,
                            const UnidirectionalSequenceRNNParams &params);

  const Tensor *input() const { return _inputs[0]; }

  const Tensor *input_weights() const { return _inputs[1]; }
  const Tensor *input_recurrent_weights() const { return _inputs[2]; }
  const Tensor *input_bias() const { return _inputs[3]; }
  const Tensor *input_hidden_state() const { return _inputs[4]; }

  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;

private:
  void check_input_tensor_dimensions(int n_input, int n_output, int n_cell, bool use_layer_norm,
                                     bool is_integer);
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCERNN_H
