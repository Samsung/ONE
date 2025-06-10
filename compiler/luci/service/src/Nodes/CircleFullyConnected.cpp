/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Service/CircleShapeInference.h"

#include "CircleCloneNode.h"
#include "CircleShapeInferenceHelper.h"

#include "Check.h"

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::DEF>::visit(const luci::CircleFullyConnected *node)
{
  if (node->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return nullptr;
  if (node->weights_format() == luci::CircleFullyConnected::WeightsFormat::UNDEFINED)
    return nullptr;

  auto *cloned = _graph->nodes()->create<luci::CircleFullyConnected>();
  {
    cloned->fusedActivationFunction(node->fusedActivationFunction());
    cloned->weights_format(node->weights_format());
    cloned->keep_num_dims(node->keep_num_dims());
  }
  return cloned;
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleFullyConnected *node)
{
  auto input_shape = circle_shape(loco::must_cast<CircleNode *>(node->input()));
  auto weights_shape = circle_shape(loco::must_cast<CircleNode *>(node->weights()));

  loco::TensorShape out_shape;

  // NOTE Some recipes in some repositories are using rank 4 input for FullyConnected.
  //      Until they are all fixed, disable following assert.
  // TODO Enable following assert after related fixes are applied
  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L194
  // LUCI_ASSERT(input_shape.rank() == 2 || input_shape.rank() == 3,
  //             "Input rank of FullyConnected should be 2 or 3");

  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L225
  LUCI_ASSERT(weights_shape.rank() == 2, "Weights of FullyConnected should be 2");
  LUCI_ASSERT(weights_shape.dim(0).known() && weights_shape.dim(1).known(),
              "Weights of FullyConnected should be known")
  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L353-L367

  /*
   * **Pre-conditions:**
   *    input_shape.rank() <= 4
   *      * remark: TFLite allows <=3 ranks, but there are rank 4 input recipes in ONE
   *    weights_shape.rank() == 2 and all dimensions are known.
   *    When runtime(input_shape[-1] and weights_shape[-1] are both known), it should be same value.
   *
   * **Shape Inference Rule:**
   *    **Input Shape:**
   *      input_shape : (A, B, C, D)
   *      weights_shape : (E, F)
   *      A, B, C, D are "positive numbers" or "unknown".
   *      E, F are always "positive numbers".
   *
   *    **Output Shape:**
   *      If keep_dims = True : (A, B, C, E)
   *      If keep_dims = False : (G, E)
   *       * G = unknown (if any of A, B, or C is unknown.)
   *       * G = A * B * C  (otherwise.)
   */

  if (node->keep_num_dims())
  {
    out_shape.rank(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i)
      out_shape.dim(i) = input_shape.dim(i);
    out_shape.dim(out_shape.rank() - 1) = weights_shape.dim(0);
  }
  else
  {
    bool is_dynamic_shape = false;

    for (uint32_t i = 0; i < input_shape.rank() - 1; i++)
    {
      if (not input_shape.dim(i).known())
      {
        is_dynamic_shape = true;
        break;
      }
    }

    uint32_t batch_size = 1;

    for (uint32_t i = 0; i < input_shape.rank() - 1; i++)
    {
      batch_size *= input_shape.dim(i).value();
    }

    out_shape.rank(2);
    if (is_dynamic_shape)
      out_shape.dim(0).unset();
    else
      out_shape.dim(0) = batch_size;
    out_shape.dim(1) = weights_shape.dim(0);
  }

  return out_shape;
}

} // namespace sinf
} // namespace luci
