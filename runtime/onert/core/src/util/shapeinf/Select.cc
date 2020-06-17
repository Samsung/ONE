/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

bool HaveSameShapes(const ir::Shape &input_cond_shape, const ir::Shape &input_true_shape,
                    const ir::Shape &input_false_shape)
{
  if ((input_cond_shape.rank() != input_true_shape.rank()) ||
      input_cond_shape.rank() != input_false_shape.rank())
  {
    return false;
  }

  int rank = input_cond_shape.rank();
  for (int i = 0; i < rank; ++i)
  {
    if (input_cond_shape.dim(i) != input_true_shape.dim(i) ||
        input_cond_shape.dim(i) != input_false_shape.dim(i))
    {
      return false;
    }
  }

  return true;
}

bool calculateShape(const ir::Shape &input_cond_shape, const ir::Shape &input_true_shape,
                    const ir::Shape &input_false_shape, ir::Shape &new_shape)
{
  ir::Shape cond_shape = input_cond_shape;
  ir::Shape true_shape = input_true_shape;
  ir::Shape false_shape = input_false_shape;
  int most_rank =
      (cond_shape.rank() >= true_shape.rank()) && (cond_shape.rank() >= false_shape.rank())
          ? cond_shape.rank()
          : (false_shape.rank() >= true_shape.rank() ? false_shape.rank() : true_shape.rank());

  ir::Shape calculate_shape(most_rank);

  cond_shape.extendRank(most_rank);
  true_shape.extendRank(most_rank);
  false_shape.extendRank(most_rank);

  for (int i = 0; i < most_rank; ++i)
  {
    calculate_shape.dim(i) =
        (cond_shape.dim(i) >= true_shape.dim(i)) && (cond_shape.dim(i) >= false_shape.dim(i))
            ? cond_shape.dim(i)
            : (false_shape.dim(i) >= true_shape.dim(i) ? false_shape.dim(i) : true_shape.dim(i));

    if ((cond_shape.dim(i) != calculate_shape.dim(i) && cond_shape.dim(i) != 1) ||
        (true_shape.dim(i) != calculate_shape.dim(i) && true_shape.dim(i) != 1) ||
        (false_shape.dim(i) != calculate_shape.dim(i) && false_shape.dim(i) != 1))
    {
      return false;
    }
  }

  new_shape = calculate_shape;

  return true;
}

ir::Shape inferSelectShape(const ir::Shape &input_cond_shape, const ir::Shape &input_true_shape,
                           const ir::Shape &input_false_shape)
{
  bool havesame = HaveSameShapes(input_cond_shape, input_true_shape, input_false_shape);
  if (havesame)
  {
    return input_cond_shape;
  }

  ir::Shape new_shape;
  bool possible = calculateShape(input_cond_shape, input_true_shape, input_false_shape, new_shape);

  if (!possible)
  {
    throw std::runtime_error("Broadcasting is not possible.");
  }

  return new_shape;
}

void StaticInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx{op.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto &input_cond = _operands.at(input_cond_idx);

  const auto input_true_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto &input_true = _operands.at(input_true_idx);

  const auto input_false_idx{op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  const auto &input_false = _operands.at(input_false_idx);

  auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input_cond.info().isDynamic() || input_true.info().isDynamic() ||
      input_false.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // Select output shpae
  ir::Shape new_shape = inferSelectShape(input_cond.info().shape(), input_true.info().shape(),
                                         input_false.info().shape());
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx = op.getInputs().at(ir::operation::Select::Input::CONDITION);
  const auto &input_cond = _tensor_registry->getITensor(input_cond_idx);

  const auto input_true_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_TRUE);
  const auto &input_true = _tensor_registry->getITensor(input_true_idx);

  const auto input_false_idx = op.getInputs().at(ir::operation::Select::Input::INPUT_FALSE);
  const auto &input_false = _tensor_registry->getITensor(input_false_idx);

  if ((!input_cond->is_dynamic()) && (!input_true->is_dynamic()) && (!input_false->is_dynamic()))
  {
    return;
  }

  auto input_cond_shape = input_cond->getShape();
  auto input_true_shape = input_true->getShape();
  auto input_false_shape = input_false->getShape();

  // Select output shpae
  ir::Shape new_shape = inferSelectShape(input_cond_shape, input_true_shape, input_false_shape);

  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);

  _dynamic_tensor_manager->applyShape(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
