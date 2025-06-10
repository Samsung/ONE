/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "cker/Shape.h"
#include <cker/operation/RoPE.h>

#include <gtest/gtest.h>
#include <vector>

using nnfw::cker::Shape;
using nnfw::cker::RoPEMode;

TEST(CKer_Operation, RoPE)
{
  // float
  {
    RoPEMode mode = RoPEMode::kGptNeox;

    Shape input_shape{1, 1, 1, 4};
    std::vector<float> input{0, 1.0, 2.0, 3.0};

    Shape sin_table_shape{1, 1, 1, 4};
    std::vector<float> sin_table{0.5, 1.0, 1.0, 0.5};
    Shape cos_table_shape{1, 1, 1, 4};
    std::vector<float> cos_table{1.0, 0.5, 0.5, 1.0};

    Shape ref_output_shape{1, 1, 1, 4};
    std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

    Shape output_shape{1, 1, 1, 4};
    std::vector<float> output(ref_output_data.size());

    nnfw::cker::RoPE<float>(mode, input_shape, input.data(), sin_table_shape, sin_table.data(),
                            cos_table_shape, cos_table.data(), ref_output_shape, output.data());

    for (size_t i = 0; i < ref_output_data.size(); ++i)
    {
      EXPECT_FLOAT_EQ(ref_output_data[i], output[i]);
    }
  }

  // int64_t
  {
    RoPEMode mode = RoPEMode::kGptNeox;

    Shape input_shape{1, 1, 1, 4};
    std::vector<int64_t> input{0, 1, 2, 3};

    Shape sin_table_shape{1, 1, 1, 4};
    std::vector<int64_t> sin_table{0, 1, 1, 0};
    Shape cos_table_shape{1, 1, 1, 4};
    std::vector<int64_t> cos_table{1, 0, 0, 1};

    Shape ref_output_shape{1, 1, 1, 4};
    std::vector<int64_t> ref_output_data{0, -3, 0, 3};

    Shape output_shape{1, 1, 1, 4};
    std::vector<int64_t> output(ref_output_data.size());

    nnfw::cker::RoPE<int64_t>(mode, input_shape, input.data(), sin_table_shape, sin_table.data(),
                              cos_table_shape, cos_table.data(), ref_output_shape, output.data());

    for (size_t i = 0; i < ref_output_data.size(); ++i)
    {
      EXPECT_EQ(ref_output_data[i], output[i]);
    }
  }
}

TEST(CKer_Operation, neg_RoPE)
{
  // the dimension(3) of sin_table and input do not match
  {
    RoPEMode mode = RoPEMode::kGptNeox;

    Shape input_shape{1, 1, 1, 4};
    std::vector<float> input{0, 1.0, 2.0, 3.0};

    Shape sin_table_shape{1, 1, 1, 3};
    std::vector<float> sin_table{0.5, 1.0, 1.0};
    Shape cos_table_shape{1, 1, 1, 4};
    std::vector<float> cos_table{1.0, 0.5, 0.5, 1.0};

    Shape ref_output_shape{1, 1, 1, 4};
    std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

    std::vector<float> output(ref_output_data.size());
    Shape output_shape{1, 1, 1, 4};

    EXPECT_ANY_THROW(nnfw::cker::RoPE<float>(mode, input_shape, input.data(), sin_table_shape,
                                             sin_table.data(), cos_table_shape, cos_table.data(),
                                             ref_output_shape, output.data()));
  }

  // the dimension(3) of cos_table and input do not match
  {
    RoPEMode mode = RoPEMode::kGptNeox;

    Shape input_shape{1, 1, 1, 4};
    std::vector<float> input{0, 1.0, 2.0, 3.0};

    Shape sin_table_shape{1, 1, 1, 4};
    std::vector<float> sin_table{0.5, 1.0, 1.0, 0.5};
    Shape cos_table_shape{1, 1, 1, 3};
    std::vector<float> cos_table{1.0, 0.5, 0.5};

    Shape ref_output_shape{1, 1, 1, 4};
    std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

    std::vector<float> output(ref_output_data.size());
    Shape output_shape{1, 1, 1, 4};

    EXPECT_ANY_THROW(nnfw::cker::RoPE<float>(mode, input_shape, input.data(), sin_table_shape,
                                             sin_table.data(), cos_table_shape, cos_table.data(),
                                             ref_output_shape, output.data()));
  }

  // unsupported RoPE Mode
  {
    RoPEMode mode = RoPEMode::kGptJ;

    Shape input_shape{1, 1, 1, 4};
    std::vector<float> input{0, 1.0, 2.0, 3.0};

    Shape sin_table_shape{1, 1, 1, 4};
    std::vector<float> sin_table{0.5, 1.0, 1.0, 0.5};
    Shape cos_table_shape{1, 1, 1, 4};
    std::vector<float> cos_table{1.0, 0.5, 0.5, 1.0};

    Shape ref_output_shape{1, 1, 1, 4};
    std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

    Shape output_shape{1, 1, 1, 4};
    std::vector<float> output(ref_output_data.size());

    EXPECT_ANY_THROW(nnfw::cker::RoPE<float>(mode, input_shape, input.data(), sin_table_shape,
                                             sin_table.data(), cos_table_shape, cos_table.data(),
                                             ref_output_shape, output.data()));
  }

  // unsupported odd number
  {
    RoPEMode mode = RoPEMode::kGptNeox;

    Shape input_shape{1, 1, 1, 3};
    std::vector<float> input{0, 1.0, 2.0};

    Shape sin_table_shape{1, 1, 1, 3};
    std::vector<float> sin_table{0.5, 1.0, 1.0};
    Shape cos_table_shape{1, 1, 1, 3};
    std::vector<float> cos_table{1.0, 0.5, 0.5};

    Shape ref_output_shape{1, 1, 1, 3};
    std::vector<float> ref_output_data{-1.0, -2.5, 1.0};

    Shape output_shape{1, 1, 1, 3};
    std::vector<float> output(ref_output_data.size());

    EXPECT_ANY_THROW(nnfw::cker::RoPE<float>(mode, input_shape, input.data(), sin_table_shape,
                                             sin_table.data(), cos_table_shape, cos_table.data(),
                                             ref_output_shape, output.data()));
  }
}
