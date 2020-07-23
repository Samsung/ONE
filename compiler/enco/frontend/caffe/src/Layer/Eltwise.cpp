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

#include "Eltwise.h"
#include "IRBuilder.h"

#include <coco/IR/FeatureLayouts.h>

#include <morph/caffe.h>

#include <cassert>
#include <functional>

using namespace nncc::core::ADT;
using namespace morph::caffe;

namespace caffeimport
{

void EltwiseBuilder::build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  coco::Block *blk = context->block();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();

  using coco::FeatureLayouts::BCHW;

  assert(layer.bottom().size() > 1);
  assert(layer.top().size() == 1);

  assert(layer.has_eltwise_param());
  const auto &param = layer.eltwise_param();

  using ::caffe::EltwiseParameter_EltwiseOp;
  using ::caffe::EltwiseParameter_EltwiseOp_PROD;
  using ::caffe::EltwiseParameter_EltwiseOp_SUM;

  using Reducer = std::function<coco::Op *(coco::Op * lhs, coco::Op * rhs)>;
  using ReducerRegistry = std::map<EltwiseParameter_EltwiseOp, Reducer>;

  ReducerRegistry registry;

  // MAX are not supported, yet
  registry[EltwiseParameter_EltwiseOp_SUM] = [](coco::Op *lhs, coco::Op *rhs) -> coco::Op * {
    if (lhs == nullptr)
    {
      assert(rhs != nullptr);
      return rhs;
    }

    assert(lhs != nullptr && rhs != nullptr);
    assert(lhs->module() == rhs->module());
    assert(lhs->module() != nullptr);

    auto m = lhs->module();
    return op_builder(m).push(rhs).push(lhs).add().pop();
  };

  registry[EltwiseParameter_EltwiseOp_PROD] = [](coco::Op *lhs, coco::Op *rhs) -> coco::Op * {
    if (lhs == nullptr)
    {
      assert(rhs != nullptr);
      return rhs;
    }

    assert(lhs != nullptr && rhs != nullptr);
    assert(lhs->module() == rhs->module());
    assert(lhs->module() != nullptr);

    auto m = lhs->module();
    return op_builder(m).push(rhs).push(lhs).mul().pop();
  };

  // coeff is not supported, yet
  assert(!param.coeff().size());

  // Decide appropriate reduce function
  auto reduce = registry.at(param.operation());

  coco::Op *op = nullptr;

  for (const auto &ifm_name : layer.bottom())
  {
    auto ifm_shape = shape_ctx.at(ifm_name);

    // NOTE The current implementation does not work in general
    auto ifm_bag = bag_ctx.at(ifm_name);
    auto ifm_obj = module->entity()->object()->create<coco::FeatureObject>();

    ifm_obj->bag(ifm_bag);
    ifm_obj->layout(BCHW::create(as_feature_shape(ifm_shape)));

    auto load = op_builder(module).load(ifm_obj).pop();

    op = reduce(op, load);
  }

  assert(op != nullptr);

  const auto ofm_name = layer.top(0);
  const auto ofm_shape = shape_ctx.at(layer.bottom(0));

  auto ofm_bag = module->entity()->bag()->create(num_elements(ofm_shape));
  auto ofm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(BCHW::create(as_feature_shape(ofm_shape)));

  // Create "Eval" instruction
  auto eval = instr_builder(module).eval(ofm_obj, op);

  // Append the instruction to the block
  blk->instr()->append(eval);

  // Update bag and shape context
  bag_ctx[ofm_name] = ofm_bag;
  shape_ctx[ofm_name] = ofm_shape;
}

} // namespace caffeimport
