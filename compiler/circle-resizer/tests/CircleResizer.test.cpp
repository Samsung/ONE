/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleResizer.h"

#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace circle_resizer;

namespace {

// TODO: Remove, just for debug
void print_shapes(const std::vector<Shape>& shapes) {
    for (const auto& shape : shapes) {
      std::cout << "[";
        for (const auto& dim : shape) {
            std::cout << dim.value() << ",";
        }
        std::cout << "],";
    }
    std::cout << std::endl;
}

} // namespace

class CircleResizerTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    _test_models_dir = std::getenv("ARTIFACTS_PATH");
    assert(!_test_models_dir.empty());
  }
protected:
    std::string _test_models_dir;
};

TEST_F(CircleResizerTest, basic_test)
{
    CircleResizer resizer(_test_models_dir + "/DynInputs_Add_001.circle");
    const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{1}}, Shape{Dim{1}, Dim{5}, Dim{1}}};
    resizer.resize_model(new_input_shapes);
    ASSERT_EQ(resizer.input_shapes(), new_input_shapes);
    ASSERT_EQ(resizer.output_shapes(), (std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{1}}}));
}
