/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tfinfo-v2/TensorInfoLoader.h"

#include "tfinfo-v2/TensorSignature.h"

#include <gtest/gtest.h>

#include <sstream>
#include <map>

#define TC_CASE(content) #content

using namespace tfinfo::v2;

namespace
{

// clang-format off
const std::vector<std::string> success_cases =
{
    TC_CASE(
                output {
                    name : "relu:0"
                }
    ),

    TC_CASE(
                input {
                    name : "placeholder:0"
                }

                input {
                    name : "placeholder:1"
                    dim { axis:0 size: 1 }
                    dim { axis:2 size: 4 }
                }

                output {
                    name : "relu:0"
                }
    ),
  // clang-format on
};

} // namespace

TEST(TFINFO_V2, success_0)
{
  std::stringstream ss{success_cases[0]};

  auto tensors = load(&ss, "tfinfo_v2_test");

  std::map<std::string, tfinfo::v2::TensorSignature *> m;

  for (auto &tensor : tensors)
  {
    m[tensor->name()] = tensor.get();
  }

  ASSERT_EQ(m.size(), 1);

  auto t1 = m["relu:0"];
  {
    ASSERT_EQ(t1->kind(), tfinfo::v2::TensorSignature::Kind::Output);
    ASSERT_TRUE(t1->shapeHint().empty());
  }
}

TEST(TFINFO_V2, success_1)
{
  std::stringstream ss{success_cases[1]};

  auto tensors = load(&ss, "tfinfo_v2_test");

  std::map<std::string, tfinfo::v2::TensorSignature *> m;

  for (auto &tensor : tensors)
  {
    m[tensor->name()] = tensor.get();
  }

  ASSERT_EQ(m.size(), 3);

  auto t1 = m["placeholder:0"];
  {
    ASSERT_EQ(t1->kind(), tfinfo::v2::TensorSignature::Kind::Input);
    ASSERT_TRUE(t1->shapeHint().empty());
  }

  auto t2 = m["placeholder:1"];
  {
    ASSERT_EQ(t2->kind(), tfinfo::v2::TensorSignature::Kind::Input);
    ASSERT_FALSE(t2->shapeHint().empty());

    auto iter = t2->shapeHint().cbegin();

    ASSERT_TRUE(iter != t2->shapeHint().cend());
    ASSERT_EQ(iter->first, 0);  // axis
    ASSERT_EQ(iter->second, 1); // size

    iter++;

    ASSERT_TRUE(iter != t2->shapeHint().cend());
    ASSERT_EQ(iter->first, 2);  // axis
    ASSERT_EQ(iter->second, 4); // size

    iter++;

    ASSERT_TRUE(iter == t2->shapeHint().cend());
  }

  auto t3 = m["relu:0"];
  {
    ASSERT_EQ(t3->kind(), tfinfo::v2::TensorSignature::Kind::Output);
    ASSERT_TRUE(t3->shapeHint().empty());
  }
}

namespace
{

// clang-format off
const std::vector<std::string> fail_cases =
  {
    // no output
    TC_CASE(
                input {
                    name : "relu:0"
                }
    ),

    // no name in input
    TC_CASE(
                input {
                    shape {
                      dim { size: 1 }
                      dim { size: 2 }
                    }
                }
                output {
                    name : "relu:0"
                }
    ),

    // wrong name format - no tensor index
    TC_CASE(
                output {
                    name : "name_with_no_index"
                }
    ),

    // wrong name format - no name but numbers
    TC_CASE(
                output {
                    name : "1"
                }
    ),

    // duplicated node def - input, input
    TC_CASE(
                input {
                    name : "duplicated_name:0"
                }

                input {
                    name : "duplicated_name:0"
                }
    ),

    // duplicated node def - input, output
    TC_CASE(
                input {
                    name : "duplicated_name:0"
                }

                output {
                    name : "duplicated_name:0"
                }
    ),

    // wrong keyword ('in', 'out' instead of 'input', 'output')
    TC_CASE(
                in {
                    name : "a:0"
                }

                out {
                    name : "b:0"
                }
    ),

    // wrong keyword ('input_name' instead of 'name')
    TC_CASE(
                input {
                    input_name : "a:0"
                }

                output {
                    name : "b:0"
                }
    ),

    // using deprecated format
    // (note that because of ",", macro TC_CASE cannot be used.)
    R"(
                input, a:0, TF_FLOAT, [2, 3 ,4]
                output, b:0, TF_FLOAT, [2, 3 ,4]
      )",
  // clang-format on
};

} // namespace

TEST(TFINFO_V2, failure)
{
  for (int i = 0; i < fail_cases.size(); i++)
  {
    std::stringstream ss{fail_cases[i]};

    try
    {
      load(&ss, "tfinfo_v2_test");

      FAIL();
    }
    catch (const std::exception &e)
    {
      std::cerr << ss.str() << std::endl << e.what() << '\n';
    }
  }
}
