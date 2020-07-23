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

#include "NodeExecution.h"

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Overlay.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include "nncc/core/ADT/tensor/IndexEnumerator.h"

#include <gtest/gtest.h>

namespace
{
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::make_overlay;
using nncc::core::ADT::tensor::Shape;

template <typename T>
void run_test(const T *lhs, const T *rhs, const T *expected_output, const Shape &lhs_shape,
              const Shape &rhs_shape, const Shape &out_shape, loco::DataType expected_datatype)
{
  auto g = loco::make_graph();
  // Fill lhs MatrixEncode
  auto lhs_enc = g->nodes()->create<loco::MatrixEncode>();
  {
    auto lhs_enc_buf = make_buffer<T, LexicalLayout>(lhs_shape);
    auto lhs_overlay = make_overlay<T, LexicalLayout>(lhs_shape, const_cast<T *>(lhs));
    for (nncc::core::ADT::tensor::IndexEnumerator e{lhs_shape}; e.valid(); e.advance())
    {
      const auto &ind = e.current();
      lhs_enc_buf.at(ind) = lhs_overlay.at(ind);
    }

    auto enc_data = locomotiv::make_data(lhs_enc_buf);
    locomotiv::annot_data(lhs_enc, std::move(enc_data));
    locomotiv::annot_domain(lhs_enc, loco::Domain::Matrix);
  }
  // Fill rhs MatrixEncode
  auto rhs_enc = g->nodes()->create<loco::MatrixEncode>();
  {
    auto rhs_enc_buf = make_buffer<T, LexicalLayout>(rhs_shape);
    auto rhs_overlay = make_overlay<T, LexicalLayout>(rhs_shape, const_cast<T *>(rhs));
    for (nncc::core::ADT::tensor::IndexEnumerator e{rhs_shape}; e.valid(); e.advance())
    {
      const auto &ind = e.current();
      rhs_enc_buf.at(ind) = rhs_overlay.at(ind);
    }

    auto enc_data = locomotiv::make_data(rhs_enc_buf);
    locomotiv::annot_data(rhs_enc, std::move(enc_data));
    locomotiv::annot_domain(rhs_enc, loco::Domain::Matrix);
  }

  // build MatMul
  auto mat_mul = g->nodes()->create<loco::MatMul>();
  mat_mul->lhs(lhs_enc);
  mat_mul->rhs(rhs_enc);

  // run interpreter
  locomotiv::NodeExecution::get().run(mat_mul);

  // get result of calculation
  auto mat_mul_result = locomotiv::annot_data(mat_mul);

  // check the result
  ASSERT_NE(mat_mul_result, nullptr);
  ASSERT_TRUE(mat_mul_result->dtype() == expected_datatype);
  ASSERT_TRUE(*(mat_mul_result->shape()) == out_shape);

  auto out_overlay = make_overlay<T, LexicalLayout>(out_shape, const_cast<T *>(expected_output));
  for (nncc::core::ADT::tensor::IndexEnumerator e{out_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    if (expected_datatype == loco::DataType::FLOAT32)
      ASSERT_FLOAT_EQ(out_overlay.at(ind), mat_mul_result->as_f32_bufptr()->at(ind));
    else if (expected_datatype == loco::DataType::S32)
      ASSERT_EQ(out_overlay.at(ind), mat_mul_result->as_s32_bufptr()->at(ind));
    else
      throw std::runtime_error("NYI for these DataTypes");
  }

  ASSERT_EQ(loco::Domain::Matrix, locomotiv::annot_domain(mat_mul));
}

} // namespace

// clang-format off
/* from the code below:

import numpy as np

a = [[-0.48850584,  1.4292705,  -1.3424522],
     [1.7021934,  -0.39246717,  0.6248314]]

b = [[-0.0830195,  0.21088193, -0.11781317],
     [0.07755677, 1.6337638,   1.0792778],
     [-1.6922939, -1.5437212,   0.96667504]]

print(np.array(a) @ np.array(b))
*/
TEST(NodeExecution_MatMul, f32_2x3_3x3)
{
  using nncc::core::ADT::tensor::Shape;

  const float lhs[] =
  {
    -0.48850584,  1.4292705,  -1.3424522,
     1.7021934,  -0.39246717,  0.6248314
  };

  const float rhs[] =
  {
    -0.0830195,  0.21088193, -0.11781317,
     0.07755677, 1.6337638,   1.0792778,
    -1.6922939, -1.5437212,   0.96667504
  };

  const float out[] =
  {
    2.42322878,  4.30444527,  0.30241731,
    -1.2291521,  -1.2468023,  -0.02011299
  };

  run_test<float>(lhs, rhs, out, Shape{2, 3}, Shape{3, 3}, Shape{2, 3}, loco::DataType::FLOAT32);
}

/* from the code below:

import numpy as np

a = np.random.randint(10000, size=(4, 2))

b = np.random.randint(10000, size=(2, 6))

print(a)
print(b)
print(np.array(a) @ np.array(b))
*/
TEST(NodeExecution_MatMul, s32_4x2_2x6)
{
  using nncc::core::ADT::tensor::Shape;

  const int32_t lhs[] =
  {
    6392, 4993,
      54, 9037,
    3947, 5820,
    5800, 4181
  };

  const int32_t rhs[] =
  {
    2694, 8376, 8090, 1285, 7492, 1652,
    5427, 8798, 7634, 2229, 5439, 6999
  };

  const int32_t out[] =
  {
    44317059, 97467806, 89827842, 19343117, 75045791, 45505591,
    49189275, 79959830, 69425318, 20212863, 49556811, 63339171,
    42218358, 84264432, 76361110, 18044675, 61225904, 47254624,
    38315487, 85365238, 78839754, 16772449, 66194059, 38844419
  };

  run_test<int32_t>(lhs, rhs, out, Shape{4, 2}, Shape{2, 6}, Shape{4, 6}, loco::DataType::S32);
}

// clang-format on
