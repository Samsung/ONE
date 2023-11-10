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

#include <cker/eigen/Utils.h>
#include <cker/operation/MaxPool.h>
#include <cker/train/operation/MaxPool.h>
#include <cker/Shape.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{
using namespace nnfw::cker;

template <typename T> class MaxPoolOpVerifier
{
private:
  const PoolParams _op_params;
  const Shape _in_shape;
  const Shape _out_shape;
  std::vector<int> _arg_max_index;

public:
  MaxPoolOpVerifier(const nnfw::cker::PoolParams &op_params, const Shape &in_shape,
                    const Shape &out_shape)
    : _op_params(op_params), _in_shape(in_shape), _out_shape(out_shape)
  {
    _arg_max_index.reserve(_out_shape.FlatSize());
  }

public:
  void verifyForward(const std::vector<T> input, const std::vector<T> expected_output,
                     bool expect_eq = true)
  {
    assert(input.size() == _in_shape.FlatSize());
    assert(expected_output.size() == _out_shape.FlatSize());

    std::vector<T> cacluated_output(_out_shape.FlatSize());
    nnfw::cker::train::MaxPool2D(_op_params, _in_shape, input.data(), _out_shape,
                                 cacluated_output.data(), _arg_max_index.data());

    if (expect_eq)
      EXPECT_EQ(expected_output, cacluated_output);
    else
      EXPECT_NE(expected_output, cacluated_output);
  }

  void verifyBackward(const std::vector<T> incoming_data, const std::vector<T> expected_grad_data,
                      bool expect_eq = true)
  {
    assert(incoming_data.size() == _out_shape.FlatSize());
    assert(expected_grad_data.size() == _in_shape.FlatSize());

    std::vector<T> calcuated_grad(_in_shape.FlatSize());
    nnfw::cker::train::MaxPool2DGrad(_out_shape, incoming_data.data(), _arg_max_index.data(),
                                     _in_shape, calcuated_grad.data());

    if (expect_eq)
      EXPECT_EQ(expected_grad_data, calcuated_grad);
    else
      EXPECT_NE(expected_grad_data, calcuated_grad);
  }
};

} // namespace

TEST(CKer_Operation, MaxPool2D)
{
  // Depth 1 case
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
    nnfw::cker::Shape in = {1, 3, 3, 1};
    nnfw::cker::Shape out = {1, 2, 2, 1};

    MaxPoolOpVerifier<float> verifier(op_param, in, out);

    /**
     *  input(index) :                         output(arg-index):
     *
     *  10(0)  15(1)   2(2)
     *   7(3)   8(4)   9(5)   - (forward) ->   15(1)  15(1)
     *  10(6)   1(7)   0(8)                    10(6)   9(5)
     */

    std::vector<float> input = {10, 15, 2, 7, 8, 9, 10, 1, 0};
    std::vector<float> expected_output = {15, 15, 10, 9};
    verifier.verifyForward(input, expected_output);

    /**
     *  output_deriv:                     input_deriv:
     * (randomly filled)
     *
     *   0.1   0.2                        0     0.3   0
     *   0.3   0.4     - (backward) ->    0     0     0.4
     *                                    0.3   0     0
     */

    std::vector<float> output_deriv = {0.1, 0.2, 0.3, 0.4};
    std::vector<float> expected_input_deriv = {0, 0.3, 0, 0, 0, 0.4, 0.3, 0, 0};
    verifier.verifyBackward(output_deriv, expected_input_deriv);
  }

  // Depth 2 case
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
    nnfw::cker::Shape in = {1, 3, 3, 2};
    nnfw::cker::Shape out = {1, 1, 1, 2};

    MaxPoolOpVerifier<float> verifier(op_param, in, out);

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
     */

    std::vector<float> input(in.FlatSize());
    auto input_mat = MapAsMatrixWithLastDimAsRows(input.data(), in);
    input_mat << /* depth0 */ 10, 15, 2, 10, 12, 17, 50, 34, -2,
      /* depth1 */ -1, 2, 3, 8, 9, 2, 4, 2, 1;
    std::vector<float> expected_output = {50, 9};
    verifier.verifyForward(input, expected_output);

    /**
     * depth[0]
     * ouput_deriv:                 input_deriv:
     *
     *                              0   0   0
     *    0.5     -(backward)->     0   0   0
     *                             0.5  0   0
     *
     *
     * depth[1]
     * output_deriv:                input_deriv:
     *                              0   0   0
     *    0.9     -(backward)->     0  0.9  0
     *                              0   0   0
     */

    std::vector<float> output_deriv = {0.5, 0.9};
    std::vector<float> expected_input_deriv(in.FlatSize());
    auto input_deriv_mat = MapAsMatrixWithLastDimAsRows(expected_input_deriv.data(), in);
    input_deriv_mat << /* depth0 */ 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
      /* depth1 */ 0, 0, 0, 0, 0.9, 0, 0, 0, 0;
    verifier.verifyBackward(output_deriv, expected_input_deriv);
  }

  // with padding case
  {
    nnfw::cker::PoolParams op_param;
    {
      op_param.stride_height = 2;
      op_param.stride_width = 2;
      op_param.filter_height = 2;
      op_param.filter_width = 2;
      op_param.padding_values.height = 2;
      op_param.padding_values.width = 2;
    }
    nnfw::cker::Shape in = {1, 2, 2, 1};
    nnfw::cker::Shape out = {1, 3, 3, 1};

    MaxPoolOpVerifier<float> verifier(op_param, in, out);

    /**
     * input_with_padding:             expected_output:
     *
     *    4   8                              0  0  0
     *    9   2            -(forward)->      0  9  0
     *                                       0  0  0
     */

    std::vector<float> input = {4, 8, 9, 2};
    std::vector<float> expected_output = {0, 0, 0, 0, 9, 0, 0, 0, 0};
    verifier.verifyForward(input, expected_output);

    /**
     * output_deriv:                    input_deriv:
     *
     *  0.1   0.1   0.1                     0     0
     *  0.1   0.2   0.3   -(backward)->     0.2   0
     *  0.5   0.1   0.1
     */
    std::vector<float> output_deriv = {0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.1};
    std::vector<float> expected_input_deriv = {0, 0, 0.2, 0};
    verifier.verifyBackward(output_deriv, expected_input_deriv);
  }
}

TEST(CKer_Operation, neg_MaxPool)
{
  // Invalid expected value
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
    nnfw::cker::Shape in = {1, 2, 2, 1};
    nnfw::cker::Shape out = {1, 1, 1, 1};

    MaxPoolOpVerifier<float> verifier(op_param, in, out);

    std::vector<float> input = {0, 0, 0, 0};
    std::vector<float> expected_output = {-1};

    verifier.verifyForward(input, expected_output, false);
  }

  // Invalid expected value
  {
    nnfw::cker::PoolParams op_param;
    {
      op_param.stride_height = 2;
      op_param.stride_width = 2;
      op_param.filter_height = 2;
      op_param.filter_width = 2;
      op_param.padding_values.height = 1;
      op_param.padding_values.width = 1;
    }

    nnfw::cker::Shape in = {1, 2, 2, 1};
    nnfw::cker::Shape out = {1, 2, 2, 1};

    MaxPoolOpVerifier<float> verifier(op_param, in, out);

    std::vector<float> input = {0, 0, 0, 0};
    std::vector<float> expected_output = {0, 0, 0, 0};
    verifier.verifyForward(input, expected_output);

    std::vector<float> output_deriv = {0.1, 0.1, 0.1, 0.2};
    std::vector<float> expected_input_deriv = {0.1, 0.1, 0.1, 0.1};
    verifier.verifyBackward(output_deriv, expected_input_deriv, false);
  }
}
