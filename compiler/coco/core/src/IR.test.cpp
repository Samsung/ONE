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

#include "coco/IR.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <nncc/core/ADT/feature/CHWLayout.h>

#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <gtest/gtest.h>

#include <set>
#include <map>
#include <string>

using nncc::core::ADT::feature::num_elements;

using nncc::core::ADT::kernel::num_elements;

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::num_elements;

//
// 'caffe_conv' test demonstrates how to translate the following Caffe network into coco IR:
//
// layer {
//   name: "data"
//   type: "Input"
//   top: "data"
//   input_param: { shape: { dim: 1 dim: 1 dim: 3 dim: 3 } }
// }
//
// layer {
//   name: "conv"
//   type: "Convolution"
//   bottom: "data"
//   top: "conv"
//   blobs {
//     ...
//     shape { dim: 1 dim: 1 dim: 3 dim: 3 }
//   }
//   convolution_param {
//     bias_term: false
//     num_output: 1
//     kernel_size: 3
//   }
// }
//
TEST(IR, caffe_conv)
{
  // For inter-layer communication
  std::map<std::string, coco::Bag *> bags;
  std::map<std::string, nncc::core::ADT::tensor::Shape> shapes;

  std::set<std::string> top_blobs;

  // Create a module and block
  auto m = coco::Module::create();
  auto blk = m->entity()->block()->create();

  // Next, append the block to the module
  m->block()->append(blk);

  // Now, the block belongs to the module (and has no sibling)
  ASSERT_EQ(blk->parent(), m.get());
  ASSERT_EQ(blk->next(), nullptr);
  ASSERT_EQ(blk->prev(), nullptr);

  // The head and tail points to the appended block
  ASSERT_EQ(m->block()->head(), blk);
  ASSERT_EQ(m->block()->tail(), blk);

  // Let's translate the first 'Input' layer
  {
    using nncc::core::ADT::tensor::Shape;

    const Shape shape{1, 1, 3, 3};

    auto bag = m->entity()->bag()->create(num_elements(shape));
    auto input = m->entity()->input()->create(shape);

    input->bag(bag);
    input->name("data");

    // Caffe uses lexical layout for tensors
    for (IndexEnumerator e{shape}; e.valid(); e.advance())
    {
      const static LexicalLayout l{};
      const auto offset = static_cast<uint32_t>(l.offset(shape, e.current()));

      input->at(e.current()) = coco::ElemID{offset};
    }

    m->input()->insert(input);

    bags["data"] = bag;
    shapes["data"] = shape;

    top_blobs = {"data"};
  }

  // Next, translate 'Convolution' layer
  {
    using nncc::core::ADT::feature::CHWLayout;
    using nncc::core::ADT::kernel::NCHWLayout;

    const nncc::core::ADT::feature::Shape ifm_shape{1, 3, 3};
    auto ifm_bag = bags["data"];
    auto ifm_obj = m->entity()->object()->create<coco::FeatureObject>();
    auto ifm_layout = coco::FeatureLayouts::BCHW::create(ifm_shape);

    ifm_obj->bag(ifm_bag);
    ifm_obj->layout(std::move(ifm_layout));

    const nncc::core::ADT::kernel::Shape ker_shape{1, 1, 3, 3};
    auto ker_bag = m->entity()->bag()->create(num_elements(ker_shape));
    auto ker_layout = coco::KernelLayouts::Generic::create(ker_shape);

    ker_layout->reorder<NCHWLayout>();

    auto ker_obj = m->entity()->object()->create<coco::KernelObject>();

    ker_obj->bag(ker_bag);
    ker_obj->layout(std::move(ker_layout));

    const nncc::core::ADT::feature::Shape ofm_shape{1, 1, 1};
    auto ofm_bag = m->entity()->bag()->create(1 * 1 * 1);
    auto ofm_obj = m->entity()->object()->create<coco::FeatureObject>();
    auto ofm_layout = coco::FeatureLayouts::BCHW::create(ifm_shape);

    ofm_obj->bag(ofm_bag);
    ofm_obj->layout(std::move(ofm_layout));

    // Create Load operation
    auto load = m->entity()->op()->create<coco::Load>();

    load->object(ifm_obj);

    // Create Conv2D operation
    //
    // NOTE Conv2D op in coco IR does not perform BiasAdd
    auto op = m->entity()->op()->create<coco::Conv2D>();

    op->ker(ker_obj);

    // Create UnitF instruction with Conv2D operation
    auto ins = m->entity()->instr()->create<coco::Eval>();

    ins->out(ofm_obj);
    ins->op(op);

    // Append the instruction (to the block)
    blk->instr()->append(ins);

    bags["conv"] = ofm_bag;
    shapes["conv"] = nncc::core::ADT::tensor::Shape{1, 1, 1, 1};

    top_blobs = {"conv"};
  }

  // Finalize
  for (const auto &top_blob : top_blobs)
  {
    const auto &shape = shapes[top_blob];

    auto output = m->entity()->output()->create(shape);

    output->bag(bags[top_blob]);
    output->name(top_blob);

    for (IndexEnumerator e{shape}; e.valid(); e.advance())
    {
      const static LexicalLayout l{};
      const auto offset = static_cast<uint32_t>(l.offset(shape, e.current()));

      output->at(e.current()) = coco::ElemID{offset};
    }

    m->output()->insert(output);
  }

  // Let's validate the constructed IR
  {
    // There is one input whose name is 'data'
    ASSERT_EQ(m->input()->size(), 1);
    ASSERT_EQ(m->input()->at(0)->name(), "data");

    // There is one output whose name is 'conv'
    ASSERT_EQ(m->output()->size(), 1);
    ASSERT_EQ(m->output()->at(0)->name(), "conv");

    ASSERT_FALSE(m->block()->empty());

    // There is one block in the module
    auto blk = m->block()->head();

    ASSERT_EQ(blk->next(), nullptr);
    ASSERT_FALSE(blk->instr()->empty());

    // There is one instruction in the block
    auto ins = blk->instr()->head();

    ASSERT_EQ(ins->next(), nullptr);

    // That instruction is 'Eval'
    // TODO Rename 'unit'
    auto unit = ins->asEval();

    ASSERT_NE(unit, nullptr);

// TODO Rewrite below test
#if 0
    // Input #0 points to IFM
    ASSERT_NE(unit->ifm(), nullptr);
    ASSERT_EQ(unit->ifm()->bag(), m->input()->at(0)->bag());
#endif

    // Output #0 points to OFM
    ASSERT_NE(unit->out(), nullptr);
    ASSERT_EQ(unit->out()->bag(), m->output()->at(0)->bag());

    // The actual operation is Conv2D
    auto conv = unit->op()->asConv2D();

    ASSERT_NE(conv, nullptr);

    // Let's check Kernel Object
    ASSERT_NE(conv->ker(), nullptr);
// TODO Rewrite below test
#if 0
    ASSERT_NE(conv->ker()->bag(), unit->ifm()->bag());
    ASSERT_NE(conv->ker()->bag(), unit->ofm()->bag());
#endif

// One may find the correspondence among Input, Output, and Objects through ElemID
// TODO Rewrite below test
#if 0
    {
      auto input_0 = m->input()->at(0);
      auto ifm = unit->ifm();

      nncc::core::ADT::tensor::Index input_index{0, 0, 2, 2};

      // Here we can check that Input(0, 0, 2, 2) corresponds to IFM(0, 2, 2)
      ASSERT_EQ(input_0->at(input_index).value(), ifm->at(0, 2, 2).value());
    }
#endif
  }
}

//
// This test demonstrates how to use 'replaceWith' method
//
TEST(IR, bag_replaceWith)
{
  auto m = coco::Module::create();

  auto bag_1 = m->entity()->bag()->create(1);
  auto bag_2 = m->entity()->bag()->create(1);

  auto obj = m->entity()->object()->create<coco::FeatureObject>();
  obj->bag(bag_1);

  auto shuffle_1 = m->entity()->instr()->create<coco::Shuffle>();
  shuffle_1->into(bag_1);

  auto shuffle_2 = m->entity()->instr()->create<coco::Shuffle>();
  shuffle_2->from(bag_1);

  ASSERT_EQ(obj->bag(), bag_1);
  ASSERT_EQ(shuffle_1->into(), bag_1);
  ASSERT_EQ(shuffle_2->from(), bag_1);

  bag_1->replaceAllDepsWith(bag_2);

  ASSERT_EQ(obj->bag(), bag_2);
  ASSERT_EQ(shuffle_1->into(), bag_1);
  ASSERT_EQ(shuffle_2->from(), bag_1);

  bag_1->replaceWith(bag_2);

  ASSERT_EQ(obj->bag(), bag_2);
  ASSERT_EQ(shuffle_1->into(), bag_2);
  ASSERT_EQ(shuffle_2->from(), bag_2);
}
