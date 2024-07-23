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

#include "FMEqualizer.h"

#include <gtest/gtest.h>

using namespace fme_apply;

TEST(FMEqualizerTest, simple)
{
  FMEqualizer fme;
  std::vector<EqualizePattern> v;

  loco::Graph g;
  EXPECT_NO_THROW(fme.equalize(&g, v));
}

TEST(FMEqualizerTest, null_graph_NEG)
{
  FMEqualizer fme;
  std::vector<EqualizePattern> v;

  EXPECT_ANY_THROW(fme.equalize(nullptr, v));
}
