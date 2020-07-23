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

#include "NodeExecution.h"

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

namespace
{
template <typename T> loco::DataType loco_dtype() { throw std::runtime_error("Not supported yet"); }
template <> loco::DataType loco_dtype<int32_t>() { return loco::DataType::S32; }
template <> loco::DataType loco_dtype<float>() { return loco::DataType::FLOAT32; }

template <typename T> const Buffer<T> *as_bufptr(const locomotiv::NodeData *data)
{
  throw std::runtime_error("Not supported yet");
}
template <> const Buffer<int32_t> *as_bufptr<int32_t>(const locomotiv::NodeData *data)
{
  return data->as_s32_bufptr();
}
template <> const Buffer<float> *as_bufptr<float>(const locomotiv::NodeData *data)
{
  return data->as_f32_bufptr();
}

template <typename T> void test()
{
  // Make pull-BiasEncode graph
  auto g = loco::make_graph();

  auto pull = g->nodes()->create<loco::Pull>();
  {
    pull->dtype(loco_dtype<T>());
    pull->shape({1});
  }

  auto bias_enc = g->nodes()->create<loco::BiasEncode>();
  {
    bias_enc->input(pull);
  }

  // Make and assign data to pull node
  auto pull_buf = make_buffer<T, LexicalLayout>(Shape{1});
  {
    pull_buf.at(Index{0}) = static_cast<T>(100);
    auto pull_data = locomotiv::make_data(pull_buf);
    locomotiv::annot_data(pull, std::move(pull_data));
    locomotiv::annot_domain(pull, loco::Domain::Tensor);
  }

  locomotiv::NodeExecution::get().run(bias_enc);

  // check
  auto bias_enc_data = locomotiv::annot_data(bias_enc);

  ASSERT_NE(bias_enc_data, nullptr);
  ASSERT_EQ(loco_dtype<T>(), bias_enc_data->dtype());
  ASSERT_EQ(Shape{1}, *(bias_enc_data->shape()));
  ASSERT_EQ(pull_buf.at(Index{0}), as_bufptr<T>(bias_enc_data)->at(Index{0}));

  ASSERT_EQ(loco::Domain::Bias, locomotiv::annot_domain(bias_enc));
}
} // namespace

TEST(NodeExecution_BiasEncode, s32) { test<int32_t>(); }

TEST(NodeExecution_BiasEncode, f32) { test<float>(); }
