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

#include "coco/IR/Module.h"

#include <gtest/gtest.h>

TEST(IR_MODULE, create)
{
  auto m = coco::Module::create();

  ASSERT_NE(m.get(), nullptr);

  coco::Module *mutable_m = m.get();
  const coco::Module *immutable_m = m.get();

  ASSERT_NE(mutable_m->entity(), nullptr);
  ASSERT_NE(immutable_m->entity(), nullptr);

  ASSERT_NE(mutable_m->entity()->bag(), nullptr);
  ASSERT_EQ(immutable_m->entity()->bag(), mutable_m->entity()->bag());

  ASSERT_NE(mutable_m->entity()->object(), nullptr);
  ASSERT_EQ(immutable_m->entity()->object(), mutable_m->entity()->object());

  ASSERT_NE(mutable_m->entity()->op(), nullptr);
  ASSERT_EQ(immutable_m->entity()->op(), mutable_m->entity()->op());

  ASSERT_NE(mutable_m->entity()->instr(), nullptr);
  ASSERT_EQ(immutable_m->entity()->instr(), mutable_m->entity()->instr());

  ASSERT_NE(mutable_m->entity()->block(), nullptr);
  ASSERT_EQ(immutable_m->entity()->block(), mutable_m->entity()->block());

  ASSERT_NE(mutable_m->entity()->input(), nullptr);
  ASSERT_EQ(immutable_m->entity()->input(), mutable_m->entity()->input());

  ASSERT_NE(mutable_m->entity()->output(), nullptr);
  ASSERT_EQ(immutable_m->entity()->output(), mutable_m->entity()->output());

  ASSERT_NE(mutable_m->block(), nullptr);
  ASSERT_EQ(immutable_m->block(), mutable_m->block());

  ASSERT_NE(mutable_m->input(), nullptr);
  ASSERT_EQ(immutable_m->input(), mutable_m->input());

  ASSERT_NE(mutable_m->output(), nullptr);
  ASSERT_EQ(immutable_m->output(), mutable_m->output());
}

TEST(IR_MODULE, append_two_blocks)
{
  auto m = coco::Module::create();

  auto blk_1 = m->entity()->block()->create();
  m->block()->append(blk_1);

  auto blk_2 = m->entity()->block()->create();
  m->block()->append(blk_2);

  ASSERT_EQ(m->block()->head(), blk_1);
  ASSERT_EQ(m->block()->tail(), blk_2);

  ASSERT_EQ(blk_1->prev(), nullptr);
  ASSERT_EQ(blk_1->next(), blk_2);

  ASSERT_EQ(blk_2->prev(), blk_1);
  ASSERT_EQ(blk_2->next(), nullptr);

  ASSERT_EQ(blk_1->index().value(), 0);
  ASSERT_EQ(blk_2->index().value(), 1);
}

TEST(IR_MODULE, append_two_instrs)
{
  auto m = coco::Module::create();

  auto blk = m->entity()->block()->create();
  auto ins_1 = m->entity()->instr()->create<coco::Eval>();
  auto ins_2 = m->entity()->instr()->create<coco::Eval>();

  blk->instr()->append(ins_1);
  blk->instr()->append(ins_2);

  ASSERT_EQ(blk->instr()->head(), ins_1);
  ASSERT_EQ(blk->instr()->tail(), ins_2);

  ASSERT_EQ(ins_1->parent(), blk);
  ASSERT_EQ(ins_1->prev(), nullptr);
  ASSERT_EQ(ins_1->next(), ins_2);

  ASSERT_EQ(ins_2->parent(), blk);
  ASSERT_EQ(ins_2->prev(), ins_1);
  ASSERT_EQ(ins_2->next(), nullptr);

  ASSERT_EQ(ins_1->index().value(), 0);
  ASSERT_EQ(ins_2->index().value(), 1);
}

TEST(IR_MODULE, iterate_constant_block)
{
  auto m = coco::Module::create();
  auto blk = m->entity()->block()->create();
  auto ins_1 = m->entity()->instr()->create<coco::Eval>();
  auto ins_2 = m->entity()->instr()->create<coco::Eval>();

  blk->instr()->append(ins_1);
  blk->instr()->append(ins_2);

  const coco::Block *immutable_blk = blk;

  ASSERT_EQ(immutable_blk->instr()->head(), ins_1);
  ASSERT_EQ(immutable_blk->instr()->head()->next(), ins_2);
}

TEST(IR_MODULE, input_as_output)
{
  // Some NN frameworks allows users to use a network input as its output.
  //
  // For example, let us consider the following Caffe network
  //
  // name: "example"
  // layer {
  //   name: "l"
  //   type: "Input"
  //   top: "data"
  //   input_param { shape: { dim: 1 dim: 1 dim: 3 dim: 3 } }
  // }
  //
  // "data" blob is the input of this network, and it is also the output of this network.
  const nncc::core::ADT::tensor::Shape shape{1, 1, 3, 3};

  auto m = coco::Module::create();
  auto bag = m->entity()->bag()->create(9);

  auto input = m->entity()->input()->create(shape);
  auto output = m->entity()->output()->create(shape);

  input->name("data");
  input->bag(bag);

  output->name("data");
  output->bag(bag);

  ASSERT_TRUE(bag->isInput());
  ASSERT_TRUE(bag->isOutput());

  output->bag(nullptr);

  ASSERT_TRUE(bag->isInput());
  ASSERT_FALSE(bag->isOutput());
}

/**
 * This test ensures that IR entities allocated via EntityManager have a correct module link
 */
TEST(IR_Module, create_entites)
{
  using namespace coco;
  using namespace nncc::core::ADT;

  auto m = Module::create();
  auto entity = m->entity();

  ASSERT_EQ(entity->bag()->create(1)->module(), m.get());
  ASSERT_EQ(entity->object()->create<coco::FeatureObject>()->module(), m.get());
  ASSERT_EQ(entity->object()->create<coco::KernelObject>()->module(), m.get());
#define OP(Name) ASSERT_EQ(entity->op()->create<Name>()->module(), m.get());
#include "coco/IR/Op.lst"
#undef OP
#define INSTR(Name)                                 \
  {                                                 \
    auto ins = entity->instr()->create<Name>();     \
    ASSERT_EQ(ins->module(), m.get());              \
    ASSERT_TRUE(coco::isa<Name>(ins));              \
    ASSERT_NE(coco::safe_cast<Name>(ins), nullptr); \
  }
#include "coco/IR/Instr.lst"
#undef INSTR
  ASSERT_EQ(entity->block()->create()->module(), m.get());
  ASSERT_EQ(entity->input()->create(tensor::Shape{1})->module(), m.get());
  ASSERT_EQ(entity->output()->create(tensor::Shape{1})->module(), m.get());
}
