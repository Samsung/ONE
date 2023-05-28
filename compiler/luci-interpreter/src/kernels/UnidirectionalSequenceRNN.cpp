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

#include "kernels/UnidirectionalSequenceRNN.h"
#include "kernels/Utils.h"
#include "PALUnidirectionalSequenceRNN.h"

namespace luci_interpreter
{
namespace kernels
{

UnidirectionalSequenceRNN::UnidirectionalSequenceRNN(
  const Tensor *input, const Tensor *input_weights, const Tensor *input_recurrent_weights,
  const Tensor *input_bias, const Tensor *input_hidden_state, Tensor *output,
  const UnidirectionalSequenceRNNParams &params)
  : KernelWithParams<UnidirectionalSequenceRNNParams>(
      {input, input_weights, input_recurrent_weights, input_bias, input_hidden_state}, {output},
      params)
{
  // Do nothing
}

void UnidirectionalSequenceRNN::configure()
{
  LUCI_INTERPRETER_CHECK(getInputTensors().size() == 5);
  LUCI_INTERPRETER_CHECK(getOutputTensors().size() == 1);

  // TODO implement
}

void UnidirectionalSequenceRNN::execute() const
{
  switch (input()->element_type())
  {
    case loco::DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type");
  }
}

void UnidirectionalSequenceRNN::evalFloat() const
{
  // TODO implement
}

} // namespace kernels
} // namespace luci_interpreter
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

#include "kernels/UnidirectionalSequenceRNN.h"
#include "kernels/Utils.h"
#include "PALUnidirectionalSequenceRNN.h"

namespace luci_interpreter
{
namespace kernels
{

UnidirectionalSequenceRNN::UnidirectionalSequenceRNN(
  const Tensor *input, const Tensor *input_weights, const Tensor *input_recurrent_weights,
  const Tensor *input_bias, const Tensor *input_hidden_state, Tensor *output,
  const UnidirectionalSequenceRNNParams &params)
  : KernelWithParams<UnidirectionalSequenceRNNParams>(
      {input, input_weights, input_recurrent_weights, input_bias, input_hidden_state}, {output},
      params)
{
  // Do nothing
}

void UnidirectionalSequenceRNN::configure()
{
  LUCI_INTERPRETER_CHECK(getInputTensors().size() == 5);
  LUCI_INTERPRETER_CHECK(getOutputTensors().size() == 1);

  // TODO implement
}

void UnidirectionalSequenceRNN::execute() const
{
  switch (input()->element_type())
  {
    case loco::DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type");
  }
}

void UnidirectionalSequenceRNN::evalFloat() const
{
  // TODO implement
}

} // namespace kernels
} // namespace luci_interpreter
