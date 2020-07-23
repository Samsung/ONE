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

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeDataImpl, as_annotation)
{
  const Shape shape{1};
  auto buf = make_buffer<float, LexicalLayout>(shape);
  buf.at(Index{0}) = 3.14f;

  std::unique_ptr<locomotiv::NodeData> data = locomotiv::make_data(buf);

  auto g = loco::make_graph();
  auto node = g->nodes()->create<loco::Pull>();

  ASSERT_EQ(nullptr, locomotiv::annot_data(node));

  // Set annotation
  locomotiv::annot_data(node, std::move(data));

  // Get annotation
  const locomotiv::NodeData *obtained = locomotiv::annot_data(node);
  ASSERT_NE(obtained, nullptr);

  ASSERT_EQ(loco::DataType::FLOAT32, obtained->dtype());
  ASSERT_EQ(shape, *(obtained->shape()));
  ASSERT_FLOAT_EQ(3.14f, obtained->as_f32_bufptr()->at(Index{0}));

  // Erase annotation
  locomotiv::erase_annot_data(node);
  ASSERT_EQ(nullptr, locomotiv::annot_data(node));
}
