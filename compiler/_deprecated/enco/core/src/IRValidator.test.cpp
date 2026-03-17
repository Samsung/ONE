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

#include "IRValidator.h"

#include "Code.h"

#include <gtest/gtest.h>

#include <array>

namespace
{

using IntList4 = std::array<int, 4>;
using IntList2 = std::array<int, 2>;

} // namespace

// The layout of ifm, ker, ofm is NHWC, pad == {top, bottom, left, right}, and stride == {vertical,
// horizontal}.
std::unique_ptr<coco::Module> get_conv2D(IntList4 ifm, IntList4 ker, IntList4 ofm, IntList4 pad,
                                         IntList2 stride)
{
  auto module = coco::Module::create();
  auto block = module->entity()->block()->create();
  auto eval = module->entity()->instr()->create<coco::Eval>();
  auto load = module->entity()->op()->create<coco::Load>();
  auto conv2D = module->entity()->op()->create<coco::Conv2D>();

  auto ifm_obj = module->entity()->object()->create<coco::FeatureObject>();
  coco::FeatureShape ifm_shape(ifm[0], ifm[3], ifm[1], ifm[2]);
  ifm_obj->layout(coco::FeatureLayouts::BHWC::create(ifm_shape));

  auto ofm_obj = module->entity()->object()->create<coco::FeatureObject>();
  coco::FeatureShape ofm_shape(ofm[0], ofm[3], ofm[1], ofm[2]);
  ofm_obj->layout(coco::FeatureLayouts::BHWC::create(ofm_shape));

  auto ker_obj = module->entity()->object()->create<coco::KernelObject>();
  nncc::core::ADT::kernel::Shape ker_shape(ker[0], ker[3], ker[1], ker[2]);
  ker_obj->layout(coco::KernelLayouts::NHWC::create(ker_shape));

  // linking entities
  module->block()->append(block);
  block->instr()->append(eval);
  eval->op(conv2D);
  eval->out(ofm_obj);
  load->object(ifm_obj);
  conv2D->ker(ker_obj);
  conv2D->arg(load);

  // param setting
  conv2D->pad()->top(pad[0]).bottom(pad[1]).left(pad[2]).right(pad[3]);
  conv2D->stride()->vertical(stride[0]).horizontal(stride[1]);

  return std::move(module);
}

TEST(IRValidatorTest, conv2D_simple)
{
  auto ifm_nhwc = IntList4{1, 3, 3, 2};
  auto ker_nhwc = IntList4{1, 1, 1, 2};
  auto ofm_nhwc = IntList4{1, 3, 3, 1};

  auto pad_tblr = IntList4{0, 0, 0, 0};
  auto stride_vh = IntList2{1, 1};

  auto module = get_conv2D(ifm_nhwc, ker_nhwc, ofm_nhwc, pad_tblr, stride_vh);
  enco::Code code{module.get(), nullptr};

  ASSERT_TRUE(enco::validate(&code));
}

TEST(IRValidatorTest, conv2D_stride_2)
{
  auto ifm_nhwc = IntList4{1, 4, 4, 3};
  auto ker_nhwc = IntList4{2, 2, 2, 3};
  auto ofm_nhwc = IntList4{1, 3, 3, 2};

  auto pad_tblr = IntList4{1, 1, 1, 1};
  auto stride_vh = IntList2{2, 2};

  auto module = get_conv2D(ifm_nhwc, ker_nhwc, ofm_nhwc, pad_tblr, stride_vh);
  enco::Code code{module.get(), nullptr};

  ASSERT_TRUE(enco::validate(&code));
}

TEST(IRValidatorTest, conv2D_output_batch_check)
{
  auto ifm_nhwc = IntList4{1, 2, 2, 2};
  auto ker_nhwc = IntList4{3, 1, 1, 2}; // expected output depth is 3
  auto ofm_nhwc = IntList4{1, 2, 2, 1}; // but 1

  auto pad_tblr = IntList4{0, 0, 0, 0};
  auto stride_vh = IntList2{1, 1};

  auto module = get_conv2D(ifm_nhwc, ker_nhwc, ofm_nhwc, pad_tblr, stride_vh);
  enco::Code code{module.get(), nullptr};

  ASSERT_FALSE(enco::validate(&code));
}

TEST(IRValidatorTest, conv2D_wrong_HW)
{
  auto ifm_nhwc = IntList4{1, 2, 2, 1};
  auto ker_nhwc = IntList4{1, 2, 2, 1};
  auto ofm_nhwc = IntList4{1, 1, 1, 1}; // HW should be 2, 2

  auto pad_tblr = IntList4{1, 1, 1, 1};
  auto stride_vh = IntList2{2, 2};

  auto module = get_conv2D(ifm_nhwc, ker_nhwc, ofm_nhwc, pad_tblr, stride_vh);
  enco::Code code{module.get(), nullptr};

  ASSERT_FALSE(enco::validate(&code));
}
