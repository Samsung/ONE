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

#include "nest/FV.h"
#include "nest/Module.h"

#include <gtest/gtest.h>

TEST(FV, var_expr)
{
  nest::Module m;

  auto var = m.var().make();

  auto fvs = nest::FV::in(var);

  ASSERT_EQ(fvs.size(), 1);
  ASSERT_NE(fvs.find(var.id()), fvs.end());
}

TEST(FV, deref_expr)
{
  nest::Module m;

  auto dom = m.domain().make({16});
  auto var = m.var().make();

  auto fvs = nest::FV::in(dom(var));

  ASSERT_EQ(fvs.size(), 1);
  ASSERT_NE(fvs.find(var.id()), fvs.end());
}

TEST(FV, add_expr)
{
  nest::Module m;

  auto v_0 = m.var().make();
  auto v_1 = m.var().make();

  auto fvs = nest::FV::in(v_0 + v_1);

  ASSERT_EQ(fvs.size(), 2);
  ASSERT_NE(fvs.find(v_0.id()), fvs.end());
  ASSERT_NE(fvs.find(v_1.id()), fvs.end());
}

TEST(FV, mul_expr)
{
  nest::Module m;

  auto v_0 = m.var().make();
  auto v_1 = m.var().make();

  nest::FV fv;

  auto fvs = nest::FV::in(v_0 * v_1);

  ASSERT_EQ(fvs.size(), 2);
  ASSERT_NE(fvs.find(v_0.id()), fvs.end());
  ASSERT_NE(fvs.find(v_1.id()), fvs.end());
}
