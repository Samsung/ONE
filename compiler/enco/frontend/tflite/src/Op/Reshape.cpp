/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Reshape.h"

#include "IRBuilder.h"
#include "GraphBuilder.h"

#include <morph/tflite.h>
#include <coco/IR/Module.h>
#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::tflite;

namespace tflimport
{

void ReshapeGraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
{
  assert(context != nullptr); // check if init(..) is called

  coco::Module *m = context->m();
  coco::Block *blk = context->block();
  TensorBags &bags = context->bags();

  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 : input feature
  // input index 1 : output shape (int32_t), (optional or not, is not clear)
  // output index 0 : output feature
  assert(opinputs.size() == 1 || opinputs.size() == 2);
  assert(opoutputs.size() == 1);

  // Note: there are actually 3 places where we can get output shape from
  // current TF lite implementation. From output operand shape, second input,
  // and ReshapeOption (new_shape). Here we use output operand shape
  int ifm_idx = opinputs.at(0);
  int ofm_idx = opoutputs.at(0);

  auto ifm_bag = bags.bag(ifm_idx);
  auto ofm_bag = bags.bag(ofm_idx);

  // TODO: move to InstrBuilder as 'shuffle_elements()'
  // Create a 1:1 shuffle instruction from ifm into ofm
  // Note: Reshape is change of shape information and there is no value change
  // in the bag itself. We implement this as just make a element wise copy of
  // the bag from input to output. So there is no need of 'reshape' operator
  auto shuffle_ins = m->entity()->instr()->create<coco::Shuffle>();
  auto num_elem = ifm_bag->size();

  assert(num_elem == ofm_bag->size());

  shuffle_ins->from(ifm_bag);
  shuffle_ins->into(ofm_bag);

  for (uint32_t n = 0; n < num_elem; ++n)
  {
    const auto from = coco::ElemID(n);
    const auto into = coco::ElemID(n);

    shuffle_ins->insert(from, into);
  }

  // Append the instruction
  blk->instr()->append(shuffle_ins);
}

} // namespace tflimport
