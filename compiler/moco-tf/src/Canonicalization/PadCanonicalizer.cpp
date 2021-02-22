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

#include "PadCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include "loco/Service/TypeInference.h"

namespace
{

bool canonicalize_pad(loco::Graph *graph, moco::TFPad *node)
{
  /**
   * @note This will replace TFPad node with Canonical TensorConstantPad
   *
   *       Before
   *                 input --- TFPad -- C
   *                 paddings --/
   *       After
   *                 paddings  ------- TFPad --
   *                                  /
   *                 input ----------- TensorConstantPad -- C
   *                 ConstGen --------/
   *       Where
   *                 input : input of TFPad
   *                 paddings : paddings of TFPad. it becomes TensorConstantPad's attribute.
   *                 C : a node that uses TFPad as an input. TFPad is disconnected from C.
   *                 ConstGen : constant value of Pad. TFPad has zero value by default.
   */

  auto pad_node = graph->nodes()->create<loco::TensorConstantPad>();

  auto constant_node = graph->nodes()->create<loco::ConstGen>();

  auto input_node = node->input();
  // TODO: support other dtype.
  assert(loco::dtype_get(input_node) == loco::DataType::FLOAT32);
  constant_node->dtype(loco::DataType::FLOAT32);
  // TODO: constant node changes to scalar when it is implemented.
  constant_node->shape({1});
  constant_node->size<loco::DataType::FLOAT32>(1);
  constant_node->at<loco::DataType::FLOAT32>(0) = 0.0f;

  auto const_paddings_node = loco::must_cast<loco::ConstGen *>(node->paddings());
  // TODO: support S64 type.
  assert(const_paddings_node->dtype() == loco::DataType::S32);
  assert(const_paddings_node->rank() == 2);
  assert(const_paddings_node->dim(1).value() == 2);

  auto padding = pad_node->padding();
  uint32_t padding_rank = const_paddings_node->dim(0).value();
  padding->rank(padding_rank);

  for (uint32_t i = 0; i < padding_rank; i++)
  {
    padding->front(i) = const_paddings_node->at<loco::DataType::S32>(i << 1);
    padding->back(i) = const_paddings_node->at<loco::DataType::S32>((i << 1) + 1);
  }

  // update connections
  pad_node->input(input_node);
  pad_node->constant(constant_node);

  // replace node
  replace(node).with(pad_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool PadCanonicalizer::transform(TFPad *node) const
{
  return canonicalize_pad(node->graph(), node);
}

} // namespace tf
} // namespace moco
