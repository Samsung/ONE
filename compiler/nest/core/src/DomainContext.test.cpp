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

#include "nest/DomainContext.h"

#include <gtest/gtest.h>

TEST(DOMAIN_CONTEXT, usecase)
{
  nest::DomainContext ctx;

  auto dom_0 = ctx.make({1, 3, 4});

  ASSERT_EQ(ctx.count(), 1);

  auto check_dom_0 = [&](void) {
    ASSERT_EQ(ctx.info(dom_0).rank(), 3);
    ASSERT_EQ(ctx.info(dom_0).dim(0), 1);
    ASSERT_EQ(ctx.info(dom_0).dim(1), 3);
    ASSERT_EQ(ctx.info(dom_0).dim(2), 4);
  };

  check_dom_0();

  auto dom_1 = ctx.make({7, 6, 2, 1});

  ASSERT_EQ(ctx.count(), 2);

  // Domain ID should be unique for each domain
  ASSERT_FALSE(dom_0.id() == dom_1.id());

  auto check_dom_1 = [&](void) {
    ASSERT_EQ(ctx.info(dom_1).rank(), 4);
    ASSERT_EQ(ctx.info(dom_1).dim(0), 7);
    ASSERT_EQ(ctx.info(dom_1).dim(1), 6);
    ASSERT_EQ(ctx.info(dom_1).dim(2), 2);
    ASSERT_EQ(ctx.info(dom_1).dim(3), 1);
  };

  // make() SHOULD NOT affect the existing domain information
  check_dom_0();
  check_dom_1();
}
