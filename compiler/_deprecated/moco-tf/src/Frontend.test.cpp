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

#include "moco/tf/Frontend.h"

#include "TestHelper.h"

#include <sstream>

#include <gtest/gtest.h>

namespace
{

// clang-format off
const char *pbtxt_000 = STRING_CONTENT(
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 4 }
      }
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
}
);
// clang-format on

} // namespace

TEST(FrontendTests, testcase_000)
{
  moco::tf::Frontend frontend;
  moco::ModelSignature signature;

  signature.add_input(moco::TensorName("Placeholder", 0));
  signature.shape("Placeholder:0", angkor::TensorShape{4});
  signature.add_output(moco::TensorName("Identity", 0));

  std::stringstream ss{pbtxt_000};

  auto graph = frontend.load(signature, &ss, moco::tf::Frontend::FileType::Text);

  ASSERT_EQ(graph->inputs()->size(), 1);
  ASSERT_EQ(graph->inputs()->at(0)->name(), "Placeholder");
  ASSERT_NE(graph->inputs()->at(0)->shape(), nullptr);
  ASSERT_EQ(graph->inputs()->at(0)->shape()->rank(), 1);
  ASSERT_EQ(graph->inputs()->at(0)->shape()->dim(0), 4);

  ASSERT_EQ(graph->outputs()->size(), 1);
  ASSERT_EQ(graph->outputs()->at(0)->name(), "Identity");
  ASSERT_NE(graph->outputs()->at(0)->shape(), nullptr);
  ASSERT_EQ(graph->outputs()->at(0)->shape()->rank(), 1);
  ASSERT_EQ(graph->outputs()->at(0)->shape()->dim(0), 4);
}
