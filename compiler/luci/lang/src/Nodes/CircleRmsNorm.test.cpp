/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleRmsNorm.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleRmsNormTest, constructor)
{
  luci::CircleRmsNorm rms_norm;

  ASSERT_EQ(luci::CircleDialect::get(), rms_norm.dialect());
  ASSERT_EQ(luci::CircleOpcode::RMS_NORM, rms_norm.opcode());

  ASSERT_EQ(nullptr, rms_norm.input());
  ASSERT_EQ(nullptr, rms_norm.gamma());
  ASSERT_FLOAT_EQ(rms_norm.epsilon(), 1e-06);
}

TEST(CircleRmsNormTest, input_NEG)
{
  luci::CircleRmsNorm rms_norm;
  luci::CircleRmsNorm node;

  rms_norm.input(&node);
  rms_norm.gamma(&node);
  ASSERT_NE(nullptr, rms_norm.input());
  ASSERT_NE(nullptr, rms_norm.gamma());

  rms_norm.input(nullptr);
  rms_norm.gamma(nullptr);
  ASSERT_EQ(nullptr, rms_norm.input());
  ASSERT_EQ(nullptr, rms_norm.gamma());
}

TEST(CircleRmsNormTest, arity_NEG)
{
  luci::CircleRmsNorm rms_norm;

  ASSERT_NO_THROW(rms_norm.arg(1));
  ASSERT_THROW(rms_norm.arg(2), std::out_of_range);
}

TEST(CircleRmsNormTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleRmsNorm rms_norm;

  TestVisitor tv;
  ASSERT_THROW(rms_norm.accept(&tv), std::exception);
}

TEST(CircleRmsNormTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleRmsNorm rms_norm;

  TestVisitor tv;
  ASSERT_THROW(rms_norm.accept(&tv), std::exception);
}
