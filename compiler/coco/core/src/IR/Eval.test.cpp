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

#include "coco/IR/Instrs.h"
#include "coco/IR/ObjectManager.h"
#include "coco/IR/OpManager.h"

#include <gtest/gtest.h>

namespace
{
class EvalTest : public ::testing::Test
{
public:
  virtual ~EvalTest() = default;

protected:
  coco::Eval *allocate(void)
  {
    auto ins = new coco::Eval{};
    _allocated.emplace_back(ins);
    return ins;
  }

private:
  std::vector<std::unique_ptr<coco::Instr>> _allocated;
};
} // namespace

TEST_F(EvalTest, constructor)
{
  auto ins = allocate();

  ASSERT_EQ(ins->out(), nullptr);
  ASSERT_EQ(ins->op(), nullptr);
}

TEST_F(EvalTest, asEval)
{
  auto ins = allocate();

  coco::Instr *mutable_ptr = ins;
  const coco::Instr *immutable_ptr = ins;

  ASSERT_NE(mutable_ptr->asEval(), nullptr);
  ASSERT_EQ(mutable_ptr->asEval(), immutable_ptr->asEval());
}
