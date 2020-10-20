/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "../MockNode.h"
#include "ir/Operations.h"

using onert::ir::Operation;
using onert::ir::OperationIndex;
using onert::ir::Operations;

TEST(ir_Operations, basic)
{
  Operations ops;
  ops.push(std::unique_ptr<Operation>(new onert_test::ir::SimpleMock({1, 2, 3, 4}, {5, 6, 7})));
  OperationIndex idx{0u};
  ASSERT_EQ(ops.at(idx).getInputs().size(), 4);
  ASSERT_EQ(ops.at(idx).getOutputs().size(), 3);
}

TEST(ir_Operations, neg_at)
{
  Operations ops;
  ops.push(std::unique_ptr<Operation>(new onert_test::ir::SimpleMock({1, 2, 3, 4}, {5, 6, 7})));
  OperationIndex idx{99u};
  EXPECT_THROW(ops.at(idx), std::out_of_range);
}
