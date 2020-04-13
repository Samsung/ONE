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

#include "Module.h"

#include <gtest/gtest.h>

TEST(ANN_IR_MODULE, constructor)
{
  ann::Module m;

  ann::Module *mutable_ptr = &m;
  const ann::Module *immutable_ptr = &m;

  ASSERT_NE(mutable_ptr->weight(), nullptr);
  ASSERT_EQ(mutable_ptr->weight(), immutable_ptr->weight());

  ASSERT_NE(mutable_ptr->operand(), nullptr);
  ASSERT_EQ(mutable_ptr->operand(), immutable_ptr->operand());

  ASSERT_NE(mutable_ptr->operation(), nullptr);
  ASSERT_EQ(mutable_ptr->operation(), immutable_ptr->operation());
}
