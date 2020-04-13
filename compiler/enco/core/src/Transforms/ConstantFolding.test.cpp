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

#include "ConstantFolding.h"
#include "Session.h"

#include <cmath>
#include <gtest/gtest.h>

namespace
{

class BinaryNetwork
{
public:
  BinaryNetwork(coco::Module *module, coco::Data *data) : _module{module}, _data{data}
  {
    // DO NOTHING
  }

  template <typename Op> void build(void);

  void fold(void)
  {
    // Execute constant folding
    enco::make_session(_module, _data);
    enco::Code code{_module, _data};
    enco::fold_constants(&code);
  }

public:
  coco::Bag *out;
  coco::Bag *lhs;
  coco::Bag *rhs;

  coco::Eval *eval;

private:
  coco::Module *_module;
  coco::Data *_data;
};

template <typename Op> void BinaryNetwork::build(void)
{
  // Create lhs bag and object
  auto lhs_bag = _module->entity()->bag()->create(12);
  auto lhs_obj = _module->entity()->object()->template create<coco::FeatureObject>();
  coco::FeatureShape lhs_shape(1, 2, 2, 3);
  lhs_obj->bag(lhs_bag);
  lhs_obj->layout(coco::FeatureLayouts::BHWC::create(lhs_shape));

  // Create rhs bag and object
  auto rhs_bag = _module->entity()->bag()->create(12);
  auto rhs_obj = _module->entity()->object()->template create<coco::FeatureObject>();
  coco::FeatureShape rhs_shape(1, 2, 2, 3);
  rhs_obj->bag(rhs_bag);
  rhs_obj->layout(coco::FeatureLayouts::BHWC::create(rhs_shape));

  // Create output bag and object
  auto output_bag = _module->entity()->bag()->create(12);
  auto output_obj = _module->entity()->object()->template create<coco::FeatureObject>();
  coco::FeatureShape ofm_shape(1, 2, 2, 3);
  output_obj->bag(output_bag);
  output_obj->layout(coco::FeatureLayouts::BHWC::create(ofm_shape));

  // Create instruction and operations
  auto block = _module->entity()->block()->create();
  auto eval = _module->entity()->instr()->template create<coco::Eval>();
  auto load_lhs = _module->entity()->op()->template create<coco::Load>();
  auto load_rhs = _module->entity()->op()->template create<coco::Load>();
  auto add_op = _module->entity()->op()->template create<Op>();

  _module->block()->append(block);
  block->instr()->append(eval);

  load_lhs->object(lhs_obj);
  load_rhs->object(rhs_obj);
  add_op->left(load_lhs);
  add_op->right(load_rhs);

  eval->op(add_op);
  eval->out(output_obj);

  // Create a handle
  this->lhs = lhs_bag;
  this->rhs = rhs_bag;
  this->out = output_bag;

  this->eval = eval;
}

} // namespace

TEST(ConstantFoldingTest, sqrt)
{
  auto module = coco::Module::create();
  auto data = coco::Data::create();

  // Create input bag and object
  auto input_bag = module->entity()->bag()->create(12);
  auto input_obj = module->entity()->object()->create<coco::FeatureObject>();
  coco::FeatureShape ifm_shape(1, 2, 2, 3);
  input_obj->bag(input_bag);
  input_obj->layout(coco::FeatureLayouts::BHWC::create(ifm_shape));

  // Create output bag and object
  auto output_bag = module->entity()->bag()->create(12);
  auto output_obj = module->entity()->object()->create<coco::FeatureObject>();
  coco::FeatureShape ofm_shape(1, 2, 2, 3);
  output_obj->bag(output_bag);
  output_obj->layout(coco::FeatureLayouts::BHWC::create(ofm_shape));

  // Insert values into input bag
  data->f32()->allocate(input_bag);
  auto input = data->f32()->weight(input_bag);
  for (uint32_t idx = 0; idx < input.size(); ++idx)
  {
    input[idx] = (float)idx;
  }

  // Create instruction and operations
  auto block = module->entity()->block()->create();
  auto eval = module->entity()->instr()->create<coco::Eval>();
  auto load = module->entity()->op()->create<coco::Load>();
  auto sqrt_op = module->entity()->op()->create<coco::Sqrt>();

  module->block()->append(block);
  block->instr()->append(eval);

  load->object(input_obj);
  sqrt_op->arg(load);

  eval->op(sqrt_op);
  eval->out(output_obj);

  // Execute constant folding
  enco::make_session(module.get(), data.get());
  enco::Code code{module.get(), data.get()};
  enco::fold_constants(&code);

  // Validate the result
  ASSERT_EQ(data->allocated(output_bag), true);
  ASSERT_EQ(eval->out(), nullptr);

  auto output = data->f32()->weight(output_bag);
  for (uint32_t idx = 0; idx < output.size(); ++idx)
  {
    ASSERT_FLOAT_EQ(output[idx], std::sqrt(input[idx]));
  }
}

TEST(ConstantFoldingTest, element_wise_add)
{
  auto module = coco::Module::create();
  auto data = coco::Data::create();

  BinaryNetwork net{module.get(), data.get()};

  // Build a network
  net.build<coco::Add>();

  // Create alises
  auto lhs_bag = net.lhs;
  auto rhs_bag = net.rhs;
  auto output_bag = net.out;
  auto eval = net.eval;

  // Insert values into lhs and rhs bag
  data->f32()->allocate(lhs_bag);
  data->f32()->allocate(rhs_bag);
  auto lhs = data->f32()->weight(lhs_bag);
  auto rhs = data->f32()->weight(rhs_bag);
  for (uint32_t idx = 0; idx < lhs.size(); ++idx)
  {
    lhs[idx] = (float)idx;
    rhs[idx] = 1.5;
  }

  // Execute constant folding
  net.fold();

  // Validate the result
  ASSERT_EQ(data->allocated(output_bag), true);
  ASSERT_EQ(eval->out(), nullptr);

  auto output = data->f32()->weight(output_bag);
  for (uint32_t idx = 0; idx < output.size(); ++idx)
  {
    ASSERT_FLOAT_EQ(output[idx], lhs[idx] + rhs[idx]);
  }
}

TEST(ConstantFoldingTest, element_wise_sub)
{
  auto module = coco::Module::create();
  auto data = coco::Data::create();

  BinaryNetwork net{module.get(), data.get()};

  // Build a network
  net.build<coco::Sub>();

  // Create alises
  auto lhs_bag = net.lhs;
  auto rhs_bag = net.rhs;
  auto output_bag = net.out;
  auto eval = net.eval;

  // Insert values into lhs and rhs bag
  data->f32()->allocate(lhs_bag);
  data->f32()->allocate(rhs_bag);
  auto lhs = data->f32()->weight(lhs_bag);
  auto rhs = data->f32()->weight(rhs_bag);
  for (uint32_t idx = 0; idx < lhs.size(); ++idx)
  {
    lhs[idx] = (float)idx;
    rhs[idx] = 1.5;
  }

  // Execute constant folding
  net.fold();

  // Validate the result
  ASSERT_EQ(data->allocated(output_bag), true);
  ASSERT_EQ(eval->out(), nullptr);

  auto output = data->f32()->weight(output_bag);
  for (uint32_t idx = 0; idx < output.size(); ++idx)
  {
    ASSERT_FLOAT_EQ(output[idx], lhs[idx] - rhs[idx]);
  }
}

TEST(ConstantFoldingTest, element_wise_mul)
{
  auto module = coco::Module::create();
  auto data = coco::Data::create();

  BinaryNetwork net{module.get(), data.get()};

  // Build a network
  net.build<coco::Mul>();

  // Create alises
  auto lhs_bag = net.lhs;
  auto rhs_bag = net.rhs;
  auto output_bag = net.out;
  auto eval = net.eval;

  // Insert values into lhs and rhs bag
  data->f32()->allocate(lhs_bag);
  data->f32()->allocate(rhs_bag);
  auto lhs = data->f32()->weight(lhs_bag);
  auto rhs = data->f32()->weight(rhs_bag);
  for (uint32_t idx = 0; idx < lhs.size(); ++idx)
  {
    lhs[idx] = (float)idx;
    rhs[idx] = 1.5;
  }

  // Execute constant folding
  net.fold();

  // Validate the result
  ASSERT_EQ(data->allocated(output_bag), true);
  ASSERT_EQ(eval->out(), nullptr);

  auto output = data->f32()->weight(output_bag);
  for (uint32_t idx = 0; idx < output.size(); ++idx)
  {
    ASSERT_FLOAT_EQ(output[idx], lhs[idx] * rhs[idx]);
  }
}

TEST(ConstantFoldingTest, element_wise_div)
{
  auto module = coco::Module::create();
  auto data = coco::Data::create();

  BinaryNetwork net{module.get(), data.get()};

  // Build a network
  net.build<coco::Div>();

  // Create alises
  auto lhs_bag = net.lhs;
  auto rhs_bag = net.rhs;
  auto output_bag = net.out;
  auto eval = net.eval;

  // Insert values into lhs and rhs bag
  data->f32()->allocate(lhs_bag);
  data->f32()->allocate(rhs_bag);
  auto lhs = data->f32()->weight(lhs_bag);
  auto rhs = data->f32()->weight(rhs_bag);
  for (uint32_t idx = 0; idx < lhs.size(); ++idx)
  {
    lhs[idx] = (float)idx;
    rhs[idx] = 1.5;
  }

  // Execute constant folding
  net.fold();

  // Validate the result
  ASSERT_EQ(data->allocated(output_bag), true);
  ASSERT_EQ(eval->out(), nullptr);

  auto output = data->f32()->weight(output_bag);
  for (uint32_t idx = 0; idx < output.size(); ++idx)
  {
    ASSERT_FLOAT_EQ(output[idx], lhs[idx] / rhs[idx]);
  }
}
