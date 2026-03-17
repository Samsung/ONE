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

#include "IR.h"

#include <gtest/gtest.h>

TEST(IRTest, ANNConv2D_default_constructor)
{
  ANNConv2D ins;

  ASSERT_EQ(ins.ofm(), nullptr);
  ASSERT_EQ(ins.ifm(), nullptr);
  ASSERT_EQ(ins.ker(), nullptr);
  ASSERT_EQ(ins.bias(), nullptr);
}

TEST(IRTest, ANNDepthConcatF_default_constructor)
{
  ANNDepthConcatF ins;

  ASSERT_EQ(ins.out(), nullptr);
  ASSERT_EQ(ins.fst(), nullptr);
  ASSERT_EQ(ins.snd(), nullptr);
}
