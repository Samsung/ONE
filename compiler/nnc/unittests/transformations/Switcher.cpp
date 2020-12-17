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

#include <gtest/gtest.h>

#include "passes/transformations/DataFormatSwitcher.h"

#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/TransposeOp.h"

TEST(TRANSFORMATIONS, Switcher_Conv2D_NCHW2NHWC)
{
  mir::Graph g;
  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 3, 299, 299}};
  auto *input = g.create<mir::ops::InputOp>(input_type);

  mir::TensorType kernel_type{mir::DataType::FLOAT32, {3, 32, 3, 3}};
  auto *kernel = g.create<mir::ops::InputOp>(kernel_type);
  // Conv2DOp
  mir::Conv2DOpAttributes attributes;
  attributes.strides = {2, 5};
  attributes.padding_before = {8, 1};
  attributes.padding_after = {7, 9};
  attributes.data_format = mir::DataFormat::NCHW;
  auto *conv = g.create<mir::ops::Conv2DOp>(input->getOutput(0), kernel->getOutput(0), attributes);

  auto *output = g.create<mir::ops::OutputOp>(conv->getOutput(0));

  nnc::DataFormatSwitcher switcher(mir::DataFormat::NHWC);

  switcher.run(&g);

  auto *trans_out = output->getInput(0)->getNode();
  auto *conv_ = trans_out->getInput(0)->getNode();
  auto *trans_in = conv_->getInput(0)->getNode();
  auto *input_ = trans_in->getInput(0)->getNode();

  ASSERT_EQ(trans_out->getType(), mir::Operation::Type::transpose);
  ASSERT_NE(conv_, conv);
  ASSERT_EQ(trans_in->getType(), mir::Operation::Type::transpose);
  ASSERT_EQ(input_, input);

  auto &in_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_in)->getAxisOrder();
  auto &out_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_out)->getAxisOrder();

  ASSERT_EQ(in_axis_order.size(), 4);
  ASSERT_EQ(in_axis_order, std::vector<size_t>({0, 2, 3, 1}));

  ASSERT_EQ(out_axis_order.size(), 4);
  ASSERT_EQ(out_axis_order, std::vector<size_t>({0, 3, 1, 2}));
  // Check Conv2D params
  auto *nhwc_conv = dynamic_cast<mir::ops::Conv2DOp *>(conv_);
  ASSERT_EQ(nhwc_conv->getDataFormat(), mir::DataFormat::NHWC);
  ASSERT_EQ(nhwc_conv->getStrides(), std::vector<int32_t>({2, 5}));
  ASSERT_EQ(nhwc_conv->getPaddingBefore(), std::vector<int32_t>({8, 1}));
  ASSERT_EQ(nhwc_conv->getPaddingAfter(), std::vector<int32_t>({7, 9}));
}

TEST(TRANSFORMATIONS, Switcher_DWConv2D_NHWC2NCHW)
{
  mir::Graph g;

  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 112, 112, 32}};
  auto *input = g.create<mir::ops::InputOp>(input_type);

  mir::TensorType kernel_type{mir::DataType::FLOAT32, {3, 3, 32, 3}};
  auto *kernel = g.create<mir::ops::InputOp>(kernel_type);
  // DepthwiseConv2DOp
  mir::Conv2DOpAttributes attributes;
  attributes.strides = {3, 25};
  attributes.padding_before = {67, 123};
  attributes.padding_after = {32, 356};
  auto *dw_conv =
    g.create<mir::ops::DepthwiseConv2DOp>(input->getOutput(0), kernel->getOutput(0), attributes);

  auto *output = g.create<mir::ops::OutputOp>(dw_conv->getOutput(0));

  nnc::DataFormatSwitcher switcher(mir::DataFormat::NCHW);

  switcher.run(&g);

  auto *trans_out = output->getInput(0)->getNode();
  auto *dw_conv_ = trans_out->getInput(0)->getNode();
  auto *trans_in = dw_conv_->getInput(0)->getNode();
  auto *input_ = trans_in->getInput(0)->getNode();

  ASSERT_EQ(trans_out->getType(), mir::Operation::Type::transpose);
  ASSERT_NE(dw_conv_, dw_conv);
  ASSERT_EQ(trans_in->getType(), mir::Operation::Type::transpose);
  ASSERT_EQ(input_, input);

  auto &in_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_in)->getAxisOrder();
  auto &out_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_out)->getAxisOrder();

  ASSERT_EQ(in_axis_order.size(), 4);
  ASSERT_EQ(in_axis_order, std::vector<size_t>({0, 3, 1, 2}));

  ASSERT_EQ(out_axis_order.size(), 4);
  ASSERT_EQ(out_axis_order, std::vector<size_t>({0, 2, 3, 1}));
  // Check DepthwiseConv2D params
  auto *nhwc_dw_conv = dynamic_cast<mir::ops::DepthwiseConv2DOp *>(dw_conv_);
  ASSERT_EQ(nhwc_dw_conv->getDataFormat(), mir::DataFormat::NCHW);
  ASSERT_EQ(nhwc_dw_conv->getStrides(), std::vector<int32_t>({3, 25}));
  ASSERT_EQ(nhwc_dw_conv->getPaddingBefore(), std::vector<int32_t>({67, 123}));
  ASSERT_EQ(nhwc_dw_conv->getPaddingAfter(), std::vector<int32_t>({32, 356}));
}

TEST(TRANSFORMATIONS, Switcher_DeConv2D_NHWC2NCHW)
{
  mir::Graph g;

  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 112, 112, 32}};
  auto *input = g.create<mir::ops::InputOp>(input_type);

  mir::TensorType kernel_type{mir::DataType::FLOAT32, {3, 3, 3, 32}};
  auto *kernel = g.create<mir::ops::InputOp>(kernel_type);
  // DeConv2DOp

  mir::Deconv2DOpAttributes attributes;
  attributes.strides = {255, 54};
  attributes.padding_before = {31, 72};
  attributes.padding_after = {32, 71};
  auto *deconv =
    g.create<mir::ops::DeConv2DOp>(input->getOutput(0), kernel->getOutput(0), attributes);

  auto *output = g.create<mir::ops::OutputOp>(deconv->getOutput(0));

  nnc::DataFormatSwitcher switcher(mir::DataFormat::NCHW);

  switcher.run(&g);

  auto *trans_out = output->getInput(0)->getNode();
  auto *deconv_ = trans_out->getInput(0)->getNode();
  auto *trans_in = deconv_->getInput(0)->getNode();
  auto *input_ = trans_in->getInput(0)->getNode();

  ASSERT_EQ(trans_out->getType(), mir::Operation::Type::transpose);
  ASSERT_NE(deconv_, deconv);
  ASSERT_EQ(trans_in->getType(), mir::Operation::Type::transpose);
  ASSERT_EQ(input_, input);

  auto &in_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_in)->getAxisOrder();
  auto &out_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_out)->getAxisOrder();

  ASSERT_EQ(in_axis_order.size(), 4);
  ASSERT_EQ(in_axis_order, std::vector<size_t>({0, 3, 1, 2}));

  ASSERT_EQ(out_axis_order.size(), 4);
  ASSERT_EQ(out_axis_order, std::vector<size_t>({0, 2, 3, 1}));
  // Check DeConv2D params
  auto *nhwc_deconv = dynamic_cast<mir::ops::DeConv2DOp *>(deconv_);
  ASSERT_EQ(nhwc_deconv->getDataFormat(), mir::DataFormat::NCHW);
  ASSERT_EQ(nhwc_deconv->getStrides(), std::vector<int32_t>({255, 54}));
  ASSERT_EQ(nhwc_deconv->getPaddingBefore(), std::vector<int32_t>({31, 72}));
  ASSERT_EQ(nhwc_deconv->getPaddingAfter(), std::vector<int32_t>({32, 71}));
}

TEST(TRANSFORMATIONS, Switcher_AvgPool2D_NHWC2NCHW)
{
  mir::Graph g;

  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 112, 112, 32}};
  auto *input = g.create<mir::ops::InputOp>(input_type);
  // AvgPool2DOp
  mir::AvgPool2DOpAttributes attributes;
  attributes.window = {41, 54};
  attributes.strides = {22, 53};
  attributes.padding_before = {11, 36};
  attributes.padding_after = {38, 45};
  auto *avg_pool = g.create<mir::ops::AvgPool2DOp>(input->getOutput(0), attributes);

  auto *output = g.create<mir::ops::OutputOp>(avg_pool->getOutput(0));

  nnc::DataFormatSwitcher switcher(mir::DataFormat::NCHW);

  switcher.run(&g);

  auto *trans_out = output->getInput(0)->getNode();
  auto *avg_pool_ = trans_out->getInput(0)->getNode();
  auto *trans_in = avg_pool_->getInput(0)->getNode();
  auto *input_ = trans_in->getInput(0)->getNode();

  ASSERT_EQ(trans_out->getType(), mir::Operation::Type::transpose);
  ASSERT_NE(avg_pool_, avg_pool);
  ASSERT_EQ(trans_in->getType(), mir::Operation::Type::transpose);
  ASSERT_EQ(input_, input);

  auto &in_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_in)->getAxisOrder();
  auto &out_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_out)->getAxisOrder();

  ASSERT_EQ(in_axis_order.size(), 4);
  ASSERT_EQ(in_axis_order, std::vector<size_t>({0, 3, 1, 2}));

  ASSERT_EQ(out_axis_order.size(), 4);
  ASSERT_EQ(out_axis_order, std::vector<size_t>({0, 2, 3, 1}));
  // Check AvgPool2D params
  auto *nhwc_avg_pool = dynamic_cast<mir::ops::AvgPool2DOp *>(avg_pool_);
  ASSERT_EQ(nhwc_avg_pool->getDataFormat(), mir::DataFormat::NCHW);
  ASSERT_EQ(nhwc_avg_pool->getWindowSize(), std::vector<int32_t>({41, 54}));
  ASSERT_EQ(nhwc_avg_pool->getStrides(), std::vector<int32_t>({22, 53}));
  ASSERT_EQ(nhwc_avg_pool->getPaddingBefore(), std::vector<int32_t>({11, 36}));
  ASSERT_EQ(nhwc_avg_pool->getPaddingAfter(), std::vector<int32_t>({38, 45}));
  ASSERT_EQ(nhwc_avg_pool->getIncludePad(), true);
}

TEST(TRANSFORMATIONS, Switcher_MaxPool2D_NCHW2NHWC)
{
  mir::Graph g;

  mir::TensorType input_type{mir::DataType::FLOAT32, {1, 3, 299, 299}};
  auto *input = g.create<mir::ops::InputOp>(input_type);

  mir::TensorType kernel_type{mir::DataType::FLOAT32, {3, 32, 3, 3}};
  auto *kernel = g.create<mir::ops::InputOp>(kernel_type);
  // MaxPool2DOp
  mir::MaxPool2DOpAttributes attributes;
  attributes.window = {41, 54};
  attributes.strides = {22, 53};
  attributes.padding_before = {11, 36};
  attributes.padding_after = {38, 45};
  attributes.data_format = mir::DataFormat::NCHW;
  auto *max_pool = g.create<mir::ops::MaxPool2DOp>(input->getOutput(0), attributes);

  auto *output = g.create<mir::ops::OutputOp>(max_pool->getOutput(0));

  nnc::DataFormatSwitcher switcher(mir::DataFormat::NHWC);

  switcher.run(&g);

  auto *trans_out = output->getInput(0)->getNode();
  auto *max_pool_ = trans_out->getInput(0)->getNode();
  auto *trans_in = max_pool_->getInput(0)->getNode();
  auto *input_ = trans_in->getInput(0)->getNode();

  ASSERT_EQ(trans_out->getType(), mir::Operation::Type::transpose);
  ASSERT_NE(max_pool_, max_pool);
  ASSERT_EQ(trans_in->getType(), mir::Operation::Type::transpose);
  ASSERT_EQ(input_, input);

  auto &in_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_in)->getAxisOrder();
  auto &out_axis_order = dynamic_cast<mir::ops::TransposeOp *>(trans_out)->getAxisOrder();

  ASSERT_EQ(in_axis_order.size(), 4);
  ASSERT_EQ(in_axis_order, std::vector<size_t>({0, 2, 3, 1}));

  ASSERT_EQ(out_axis_order.size(), 4);
  ASSERT_EQ(out_axis_order, std::vector<size_t>({0, 3, 1, 2}));
  // Check MaxPool2D params
  auto *nhwc_max_pool = dynamic_cast<mir::ops::MaxPool2DOp *>(max_pool_);
  ASSERT_EQ(nhwc_max_pool->getDataFormat(), mir::DataFormat::NHWC);
  ASSERT_EQ(nhwc_max_pool->getWindowSize(), std::vector<int32_t>({41, 54}));
  ASSERT_EQ(nhwc_max_pool->getStrides(), std::vector<int32_t>({22, 53}));
  ASSERT_EQ(nhwc_max_pool->getPaddingBefore(), std::vector<int32_t>({11, 36}));
  ASSERT_EQ(nhwc_max_pool->getPaddingAfter(), std::vector<int32_t>({38, 45}));
}
