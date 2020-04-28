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

#include "nest/Module.h"

#include <gtest/gtest.h>

TEST(MODULE, create_var)
{
  nest::Module m;

  auto create = [](nest::Module &m) {
    // This code will invoke 'VarContext &var(void)' method
    return m.var().make();
  };

  auto check = [](const nest::Module &m) {
    // This code will invoke 'const VarContext &var(void) const' method
    ASSERT_EQ(1, m.var().count());
  };

  create(m);
  check(m);
}

TEST(MODULE, create_domain)
{
  nest::Module m;

  auto create = [](nest::Module &m, std::initializer_list<uint32_t> dims) {
    // This code will invoke 'DomainContext &domain(void)' method
    return m.domain().make(dims);
  };

  auto check = [](const nest::Module &m) {
    // This code will invoke 'const DomainContext &domain(void) const' method
    ASSERT_EQ(1, m.domain().count());
  };

  create(m, {1, 3, 3});
  check(m);
}

TEST(MODULE, push)
{
  nest::Module m;

  auto ifm = m.domain().make({1, 3, 3});

  auto var_ch = m.var().make();
  auto var_row = m.var().make();
  auto var_col = m.var().make();

  m.push(ifm(var_ch, var_row, var_col));

  ASSERT_EQ(1, m.block().size());
  ASSERT_NE(m.block().at(0)->asPush(), nullptr);
}

TEST(MODULE, ret)
{
  nest::Module m;

  auto ifm = m.domain().make({1});
  auto ofm = m.domain().make({1});

  auto ind = m.var().make();

  m.push(ifm(ind));
  m.ret(ofm(ind));

  ASSERT_EQ(ofm.id(), m.ret().id());
  ASSERT_EQ(1, m.ret().sub().rank());
}

TEST(MODULE, copy)
{
  nest::Module orig;
  nest::Module copy;

  orig = copy;

  orig.var().make();

  ASSERT_EQ(0, copy.var().count());
}
