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

#include <cker/operation/ReduceMean.h>
#include <cker/train/operation/ReduceMean.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{
using namespace nnfw::cker;

template <typename T> class ReduceMeanVerifier
{
public:
  ReduceMeanVerifier(const Shape &in_shape, const Shape &out_shape,
                     const std::vector<int32_t> &axes, bool keep_dims)
    : _in_shape{in_shape}, _out_shape{out_shape}, _axes{axes}, _keep_dims{keep_dims},
      _axis_is_1_and_2{false}
  {
    _axis_is_1_and_2 = _keep_dims && _in_shape.DimensionsCount() == 4 && _axes.size() == 2 &&
                       ((_axes[0] == 1 && _axes[1] == 2) || (_axes[0] == 2 && _axes[1] == 1));
  }

  void verifyForward(const std::vector<T> &input, const std::vector<T> &expected,
                     bool expect_eq = true)
  {
    assert(input.size() == _in_shape.FlatSize());
    assert(expected.size() == _out_shape.FlatSize());

    std::vector<T> output(_out_shape.FlatSize());

    if (_axis_is_1_and_2)
    {
      nnfw::cker::MeanAxis1And2(_in_shape, input.data(), _out_shape, output.data());
    }
    else
    {
      nnfw::cker::Mean(_in_shape, input.data(), _out_shape, output.data(), _axes);
    }

    if (expect_eq)
      EXPECT_EQ(output, expected);
    else
      EXPECT_NE(output, expected);
  }

  void verifyBackward(const std::vector<T> &incoming, const std::vector<T> &expected,
                      bool expect_eq = true)
  {
    std::vector<T> grad(_in_shape.FlatSize());

    nnfw::cker::train::MeanGrad(_out_shape, incoming.data(), _in_shape, grad.data());

    if (expect_eq)
      EXPECT_EQ(grad, expected);
    else
      EXPECT_NE(grad, expected);
  }

private:
  const Shape _in_shape;
  const Shape _out_shape;
  const std::vector<int32_t> _axes;
  const bool _keep_dims;
  bool _axis_is_1_and_2;
};

} // namespace

TEST(CKer_Operation, ReduceMean)
{
  // axis = 0
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    std::vector<float> in_data = {1., 1., 2., 2.};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{1, 2};
    std::vector<float> expected_data = {1.5, 1.5};
    std::vector<int32_t> axes = {0};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyForward(in_data, expected_data);
  }

  // axis = 1
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    std::vector<float> in_data = {1., 1., 2., 2.};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{2, 1};
    std::vector<float> expected_data = {1., 2.};
    std::vector<int32_t> axes = {1};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyForward(in_data, expected_data);
  }
}

TEST(CKer_Operation, neg_ReduceMean)
{
  // wrong axis
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    std::vector<float> in_data = {1., 1., 2., 2.};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{1, 2};
    std::vector<float> expected_data = {1., 2.};
    std::vector<int32_t> axes = {0};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyForward(in_data, expected_data, false);
  }
}

TEST(CKer_Operation, ReduceMeanGrad)
{
  // axis = 0
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{1, 2};
    // nnfw::cker::Shape incoming_shape = out_shape;
    std::vector<float> incoming_data = {1., 2.};
    // nnfw::cker::Shape grad_shape = in_shape;
    std::vector<float> expected_grad_data = {0.5, 1, 0.5, 1.};
    std::vector<int32_t> axes = {0};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyBackward(incoming_data, expected_grad_data);
  }
  // axis = 1
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{2, 1};
    // nnfw::cker::Shape incoming_shape = out_shape;
    std::vector<float> incoming_data = {1., 2.};
    // nnfw::cker::Shape grad_shape = in_shape;
    std::vector<float> expected_grad_data = {0.5, 0.5, 1., 1.};
    std::vector<int32_t> axes = {1};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyBackward(incoming_data, expected_grad_data);
  }
}

TEST(CKer_Operation, neg_ReduceMeanGrad)
{
  // wrong axis
  {
    nnfw::cker::Shape in_shape = nnfw::cker::Shape{2, 2};
    nnfw::cker::Shape out_shape = nnfw::cker::Shape{1, 2};
    // nnfw::cker::Shape incoming_shape = out_shape;
    std::vector<float> incoming_data = {1., 2.};
    // nnfw::cker::Shape grad_shape = in_shape;
    std::vector<float> expected_grad_data = {0.5, 0.5, 1., 1.};
    std::vector<int32_t> axes = {0};

    ReduceMeanVerifier<float> verifier(in_shape, out_shape, axes, false /*keep_dims*/);
    verifier.verifyBackward(incoming_data, expected_grad_data, false);
  }
}
