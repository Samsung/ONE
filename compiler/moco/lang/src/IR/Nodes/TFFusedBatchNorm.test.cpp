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

#include "moco/IR/Nodes/TFFusedBatchNorm.h"
#include "moco/IR/TFDialect.h"

#include <gtest/gtest.h>

TEST(TFFusedBatchNormTest, constructor)
{
  moco::TFFusedBatchNorm fbn_node;

  ASSERT_EQ(fbn_node.dialect(), moco::TFDialect::get());
  ASSERT_EQ(fbn_node.opcode(), moco::TFOpcode::FusedBatchNorm);

  ASSERT_EQ(fbn_node.x(), nullptr);
  ASSERT_EQ(fbn_node.scale(), nullptr);
  ASSERT_EQ(fbn_node.offset(), nullptr);
  ASSERT_EQ(fbn_node.mean(), nullptr);
  ASSERT_EQ(fbn_node.variance(), nullptr);
  ASSERT_NE(fbn_node.epsilon(), 0.0f);
}
