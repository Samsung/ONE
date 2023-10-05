/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/MaxPool.h>
#include <cker/train/operation/MaxPool.h>
#include <cker/Shape.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{

struct Shape
{
  constexpr Shape(int batch, int height, int width, int depth)
    : b(batch), h(height), w(width), d(depth)
  {
  }

  int b;
  int h;
  int w;
  int d;
};

} // namespace

TEST(CKer_Operation, MaxPool2D_Depth1)
{
  nnfw::cker::PoolParams op_param;
  {
    op_param.stride_height = 1;
    op_param.stride_width = 1;
    op_param.filter_height = 2;
    op_param.filter_width = 2;
    op_param.padding_values.height = 0;
    op_param.padding_values.width = 0;
  }

  /**
   * kernel(2, 2), stride(1, 1), padding(0, 0) MaxPool forward
   *
   *
   *  input(index) :                         output(arg-index):
   *
   *  10(0)  15(1)   2(2)
   *   7(3)   8(4)   9(5)   - (forward) ->   15(1)  15(1)
   *  10(6)   1(7)   0(8)                    10(6)   9(5)
   *
   *
   *  arg_max_index(saving arguement index) :
   *
   *    1   1
   *    6   5
   *
   */

  constexpr Shape in(1, 3, 3, 1);
  constexpr Shape out(1, 2, 2, 1);

  const nnfw::cker::Shape input_shape{in.b, in.h, in.w, in.d};
  const nnfw::cker::Shape output_shape{out.b, out.h, out.w, out.d};

  std::vector<float> input = {10, 15, 2, 7, 8, 9, 10, 1, 0};
  std::vector<float> expected_output = {15, 15, 10, 9};
  std::vector<int> expected_arg_max_index = {1, 1, 6, 5};
  std::vector<float> output(out.h * out.w, 0);
  std::vector<int> arg_max_index(out.h * out.w, 0);
  nnfw::cker::train::MaxPool2D(op_param, input_shape, input.data(), output_shape, output.data(),
                               arg_max_index.data());

  ASSERT_EQ(output, expected_output);
  ASSERT_EQ(arg_max_index, expected_arg_max_index);

  /**
   * kernel(2, 2), stride(1, 1), padding(0, 0) MaxPool backward
   *
   *
   *  output_deriv:                     input_deriv:
   * (randomly filled)
   *
   *   0.1   0.2                        0     0.3   0
   *   0.3   0.4     - (backward) ->    0     0     0.4
   *                                    0.3   0     0
   */

  std::vector<float> output_deriv = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> expected_input_deriv = {0, 0.3, 0, 0, 0, 0.4, 0.3, 0, 0};
  std::vector<float> input_deriv(in.h * in.w, 0);
  nnfw::cker::train::MaxPool2DGrad(output_shape, output_deriv.data(), arg_max_index.data(),
                                   input_shape, input_deriv.data());

  ASSERT_EQ(input_deriv, expected_input_deriv);
}

TEST(CKer_Operation, MaxPool2D_Depth2)
{

  nnfw::cker::PoolParams op_param;
  {
    op_param.stride_height = 1;
    op_param.stride_width = 1;
    op_param.filter_height = 3;
    op_param.filter_width = 3;
    op_param.padding_values.height = 0;
    op_param.padding_values.width = 0;
  }

  constexpr Shape in(1, 3, 3, 2);
  constexpr Shape out(1, 1, 1, 2);

  const nnfw::cker::Shape input_shape{in.b, in.h, in.w, in.d};
  const nnfw::cker::Shape output_shape{out.b, out.h, out.w, out.d};

  /**
   *  depth[0]
   *  input(index) :                     output(arg-index):
   *
   *  10(0)  15(1)  2(2)
   *  10(3)  12(4)  17(5)   -(forward)->     50(6)
   *  50(6)  34(7)  -2(8)
   *
   *
   *  depth[1]
   *  input(index):                      output(arg-index):
   *
   *  -1(0)  2(1)  3(2)
   *  8(3)   9(4)  2(5)    -(forward)->       9(4)
   *  4(6)   2(7)  1(8)
   *
   */

  Eigen::Matrix<float, in.d, in.b * in.h * in.w, Eigen::ColMajor> input_mat;
  input_mat << /* depth0 */ 10, 15, 2, 10, 12, 17, 50, 34, -2,
    /* depth1 */ -1, 2, 3, 8, 9, 2, 4, 2, 1;

  Eigen::Matrix<float, out.d, out.b * out.h * out.w, Eigen::ColMajor> output_mat,
    expected_output_mat;
  Eigen::Matrix<int, out.d, out.b * out.h * out.w, Eigen::ColMajor> arg_max_index,
    expected_arg_max_index;
  expected_output_mat << /*depth 0*/ 50, /*depth 1*/ 9;
  expected_arg_max_index << /*depth 0*/ 6, /*depth 1*/ 4;

  nnfw::cker::train::MaxPool2D(op_param, input_shape, input_mat.data(), output_shape,
                               output_mat.data(), arg_max_index.data());

  ASSERT_EQ(output_mat, expected_output_mat);
  ASSERT_EQ(arg_max_index, expected_arg_max_index);

  /**
   * depth[0]
   * input :                      output:
   *
   *                              0   0   0
   *    0.5     -(backward)->     0   0   0
   *                             0.5  0   0
   *
   *
   * depth[1]
   * input:                       output:
   *                              0   0   0
   *    0.9     -(backward)->     0  0.9  0
   *                              0   0   0
   */

  Eigen::Matrix<float, out.d, out.b * out.h * out.w, Eigen::ColMajor> output_deriv(0.5, 0.9);
  Eigen::Matrix<float, in.d, in.b * in.h * in.w, Eigen::ColMajor> expected_input_deriv, input_deriv;
  expected_input_deriv << /* depth0 */ 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
    /* depth1 */ 0, 0, 0, 0, 0.9, 0, 0, 0, 0;
  nnfw::cker::train::MaxPool2DGrad(output_shape, output_deriv.data(), arg_max_index.data(),
                                   input_shape, input_deriv.data());

  ASSERT_EQ(input_deriv, expected_input_deriv);
}
