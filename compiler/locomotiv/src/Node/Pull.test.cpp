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
#include "UserData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

TEST(NodeExecution_Pull, check_data_ready)
{
  // Make graph with Pull node only
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();

  // Data not ready yet
  ASSERT_ANY_THROW(locomotiv::NodeExecution::get().run(pull));

  // Make and assign data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{1});
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::user_data(pull, std::move(pull_data));

// The behavior of Pull is now consistent with that of other nodes.
// -  annot_data and annot_domain is available after evaluating that "pull" node.

  // Valid run
  ASSERT_NO_THROW(locomotiv::NodeExecution::get().run(pull));
}
