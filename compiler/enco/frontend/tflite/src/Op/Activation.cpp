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

#include "Activation.h"

#include <IRBuilder.h>

#include <coco/IR/Module.h>
#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <cassert>

using namespace nncc::core::ADT;

namespace tflimport
{

coco::FeatureObject *build_activation(tflite::ActivationFunctionType act, coco::Block *block,
                                      coco::FeatureObject *ifm)
{
  assert(ifm != nullptr && ifm->asFeature() != nullptr); // support feature only in this version

  coco::Module *m = block->module();

  auto shape = ifm->asFeature()->shape();

  // creates output object
  auto output_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto output_bag = m->entity()->bag()->create(num_elements(shape));
  output_obj->bag(output_bag);
  output_obj->layout(coco::FeatureLayouts::BHWC::create(shape));

  switch (act)
  {
    case tflite::ActivationFunctionType::ActivationFunctionType_NONE:
    {
      // Create Copy Instr (copying from ifm to output_obj),
      // redundant layer but optimized by backend
      auto copy_ins = instr_builder(m).copy(output_obj, ifm);

      // Append the instruction to the block
      block->instr()->append(copy_ins);
      break;
    }
    case tflite::ActivationFunctionType::ActivationFunctionType_RELU:
    {
      // Create Eval(output_obj, ReLU(load(ifm)))
      auto load_op = op_builder(m).load(ifm).pop();
      auto relu_op = m->entity()->op()->create<coco::ReLU>();
      relu_op->arg(load_op);

      auto eval_ins = instr_builder(m).eval(output_obj, relu_op);

      // Append the instruction to the block
      block->instr()->append(eval_ins);
      break;
    }
    case tflite::ActivationFunctionType::ActivationFunctionType_RELU6:
    {
      // Create Eval(output_obj, ReLU6(load(ifm)))
      auto load_op = op_builder(m).load(ifm).pop();
      auto relu6_op = m->entity()->op()->create<coco::ReLU6>();
      relu6_op->arg(load_op);

      auto eval_ins = instr_builder(m).eval(output_obj, relu6_op);

      // Append the instruction to the block
      block->instr()->append(eval_ins);
      break;
    }
    default:
      // TODO support other fused activations
      assert(false);
      break;
  }

  return output_obj;
}

} // namespace tflimport
