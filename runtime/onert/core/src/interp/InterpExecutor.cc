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

#include "interp/InterpExecutor.h"
#include "interp/ExecEnv.h"
#include "interp/Interpreter.h"

#include "util/logging.h"

#include <memory>

namespace onert
{
namespace interp
{

void InterpExecutor::execute(const exec::IODescription &desc)
{
  /************************************************************************
   * Prepare execution model (submodel)
     It may execute divided model
     but now consider model inference is done at interpreter
   ***********************************************************************/
  ir::OperandIndexMap<std::shared_ptr<ITensor>> tensor_map;

  for (uint32_t n = 0; n < _graph.getInputs().size(); n++)
  {
    ir::IOIndex index{n};
    const auto input_index = _graph.getInputs().at(index);

    const auto input = desc.inputs.at(n).get();
    if (input == nullptr)
    {
      // Optional input
      continue;
    }

    auto input_tensor = std::make_shared<ROTensor>(input->info);
    input_tensor->setData(std::make_shared<const ir::ExternalData>(
      reinterpret_cast<const uint8_t *>(input->buffer), input->size));
    tensor_map[input_index] = input_tensor;
  }

  /************************************************************************
   * Prepare execution environment
     Execution environment will be assigned to invoked interpreter instance
   ***********************************************************************/

  std::unique_ptr<ExecEnv> interp_env = std::make_unique<ExecEnv>(_graph);

  // Assign input/output tensor into interpreter execution environment
  for (auto index : _graph.getInputs())
  {
    if (tensor_map.find(index) != tensor_map.end())
    {
      VERBOSE(INTERPRETER) << "Assign input tensor. operand index:" << index.value() << std::endl;
      interp_env->assignTensor(index, tensor_map.at(index));
    }
  }

  for (uint32_t n = 0; n < _graph.getOutputs().size(); n++)
  {
    ir::IOIndex index{n};
    const auto output_index = _graph.getOutputs().at(index);
    const auto output = desc.outputs.at(n).get();
    if (output == nullptr)
    {
      // Optional output
      continue;
    }

    VERBOSE(INTERPRETER) << "Set out buffer to ExecEnv. operand index:" << output_index.value()
                         << std::endl;

    interp_env->assignExternalBuffer(
      output_index,
      std::make_shared<ExternalBuffer>(reinterpret_cast<uint8_t *>(output->buffer), output->size));
  }

  // Allocate constant tensor
  _graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (obj.isConstant())
    {
      VERBOSE(INTERPRETER) << "Allocate and assign constant tensor. operand index:" << ind.value()
                           << std::endl;

      assert(obj.data());
      auto const_tensor = std::make_shared<ROTensor>(obj.info());
      // Assume that interpreter's tensor layout is same with model (NHWC)
      const_tensor->setData(
        std::make_shared<ir::ExternalData>(obj.data()->base(), obj.info().total_size()));
      interp_env->assignTensor(ind, const_tensor);
    }
  });

  /*****************************************************************************
   * Invoke interpreter
   ****************************************************************************/

  interp::Interpreter interp(std::move(interp_env));
  interp.run();

  /*****************************************************************************
   * Invoked interpreter run is finished
   ****************************************************************************/

  // If interpreter execute submodel
  //  1. Get tensor output of submodel into tensor_map to save result
  //  2. Generate new ExecEnv for next interpretation
}

} // namespace interp
} // namespace onert
