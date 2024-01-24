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

#include <cker/eigen/Utils.h>
#include <cker/train/operation/Pad.h>
#include <cker/Shape.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{
using namespace nnfw::cker;

template <typename T> class PadOpVerifier
{
private:
  const PadParams _op_params;
  const Shape _in_shape;
  const Shape _out_shape;
  T _constant_value;

public:
  PadOpVerifier(const PadParams &op_params, const Shape &in_shape, const Shape &out_shape, T constant_value)
    : _op_params(op_params), _in_shape(in_shape), _out_shape(out_shape), _constant_value(constant_value)
  {
    // DO NOTHING
  }

public:
  void verifyForward(const std::vector<T> input, const std::vector<T> expected_output,
                     bool expect_eq = true)
  {
    assert(input.size() == _in_shape.FlatSize());
    assert(expected_output.size() == _out_shape.FlatSize());

    std::vector<T> cacluated_output(_out_shape.FlatSize());
    nnfw::cker::train::Pad(_op_params.data, _op_params.rank,
                           _in_shape, input.data(),
                           _out_shape, cacluated_output.data(),
                           &_constant_value);

    if (expect_eq)
      EXPECT_EQ(expected_output, cacluated_output);
    else
      EXPECT_NE(expected_output, cacluated_output);
  }


  void verifyBackward(const std::vector<T> backward_output, const std::vector<T> expected_backward_input,
                      bool expect_eq = true)
  {
    assert(backward_output.size() == _out_shape.FlatSize());
    assert(expected_backward_input.size() == _in_shape.FlatSize());

    std::vector<T> backward_input(_in_shape.FlatSize());
    nnfw::cker::train::Depad(_op_params.data, _op_params.rank,
                             _out_shape, backward_output.data(),
                             _in_shape, backward_input.data());

    if (expect_eq)
      EXPECT_EQ(expected_backward_input, backward_input);
    else
      EXPECT_NE(expected_backward_input, backward_input);
  }
};

} // namespace

TEST(CKer_Operation, Pad)
{
  // NOTE:
  // Since the pad operation is an operation that only copies memory
  // (precisely where to copy memory), each input and output of forward/backward
  // can be used as each other's output and input.

  // Pad rank 1
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 1;
      op_param.data[0] = 1;
      op_param.data[1] = 1;
    }
    float constant_value = 3.f;

    nnfw::cker::Shape in = {1};
    nnfw::cker::Shape out = {3};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    std::vector<float> input = {1.f};
    std::vector<float> expected_output = {3.f,1.f,3.f};
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 1;
      op_param.data[0] = 2;
      op_param.data[1] = 2;
    }
    float constant_value = 1.f;

    nnfw::cker::Shape in = {3};
    nnfw::cker::Shape out = {7};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    std::vector<float> input = {2.f, 3.f, 4.f};
    std::vector<float> expected_output = {1.f, 1.f, 2.f, 3.f, 4.f, 1.f, 1.f};
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }

  // Pad rank 2: HW
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 2;
      op_param.data[0] = 1;
      op_param.data[1] = 1;
      op_param.data[2] = 1;
      op_param.data[3] = 1;
    }
    float constant_value = 3.f;

    nnfw::cker::Shape in = {1,1};
    nnfw::cker::Shape out = {3,3};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 1.f;
    std::vector<float> input = {init_value};
    std::vector<float> expected_output(3*3, constant_value);
    expected_output[expected_output.size()/2] = init_value;
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 2;
      op_param.data[0] = 3;
      op_param.data[1] = 3;
      op_param.data[2] = 3;
      op_param.data[3] = 3;
    }
    float constant_value = 1.f;

    nnfw::cker::Shape in = {3,3};
    nnfw::cker::Shape out = {9,9};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 1.f;
    std::vector<float> input(3*3, init_value);
    std::vector<float> expected_output(9*9, constant_value);
    for (auto i = -1; i <= 1; ++i)
    {
      for (auto j = -1; j <= 1; ++j)
      {
        size_t ind = (9 * i) + (expected_output.size()/2 + j);
        expected_output[ind] = init_value;
      }
    }
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }

  // Pad rank 3: HWC
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 3;
      op_param.data[0] = 1;
      op_param.data[1] = 1;
      op_param.data[2] = 1;
      op_param.data[3] = 1;
      op_param.data[4] = 1;
      op_param.data[5] = 1;
    }
    float constant_value = 3.f;

    nnfw::cker::Shape in = {1,1,1};
    nnfw::cker::Shape out = {3,3,3};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 1.f;
    std::vector<float> input = {init_value};
    std::vector<float> expected_output(3*3*3, constant_value);
    expected_output[expected_output.size()/2] = init_value;
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 3;
      op_param.data[0] = 5;
      op_param.data[1] = 5;
      op_param.data[2] = 5;
      op_param.data[3] = 5;
      op_param.data[4] = 5;
      op_param.data[5] = 5;
    }
    float constant_value = 7.f;

    nnfw::cker::Shape in = {3,3,3};
    nnfw::cker::Shape out = {13,13,13};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 5.f;
    std::vector<float> input(3*3*3, init_value);
    std::vector<float> expected_output(13*13*13, constant_value);
    for (auto i = -1; i <= 1; ++i)
    {
      for (auto j = -1; j <= 1; ++j)
      {
        for (auto k = -1; k <= 1; ++k)
        {
          size_t ind = (13 * 13 * i) + (13 * j) + (expected_output.size()/2 + k);
          expected_output[ind] = init_value;
        }
      }
    }
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }

  // Pad rank 4
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 4;
      op_param.data[0] = 1;
      op_param.data[1] = 1;
      op_param.data[2] = 1;
      op_param.data[3] = 1;
      op_param.data[4] = 1;
      op_param.data[5] = 1;
      op_param.data[6] = 1;
      op_param.data[7] = 1;
    }
    float constant_value = 3.f;

    nnfw::cker::Shape in = {1,1,1,1};
    nnfw::cker::Shape out = {3,3,3,3};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 1.f;
    std::vector<float> input = {init_value};
    std::vector<float> expected_output(3*3*3*3, constant_value);
    expected_output[expected_output.size()/2] = init_value;
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }
  {
    nnfw::cker::PadParams op_param;
    {
      op_param.rank = 4;
      op_param.data[0] = 7;
      op_param.data[1] = 7;
      op_param.data[2] = 7;
      op_param.data[3] = 7;
      op_param.data[4] = 7;
      op_param.data[5] = 7;
      op_param.data[6] = 7;
      op_param.data[7] = 7;
    }
    float constant_value = 9.f;

    nnfw::cker::Shape in = {5,5,5,5};
    nnfw::cker::Shape out = {19,19,19,19};

    PadOpVerifier<float> verifier(op_param, in, out, constant_value);

    float init_value = 2.f;
    std::vector<float> input(5*5*5*5, init_value);
    std::vector<float> expected_output(19*19*19*19, constant_value);
    for (auto i = -2; i <= 2; ++i)
    {
      for (auto j = -2; j <= 2; ++j)
      {
        for (auto k = -2; k <= 2; ++k)
        {
          for (auto l = -2; l <= 2; ++l)
          {
            size_t ind = (19 * 19 * 19 * i) + (19 * 19 * j) + (19 * k) + (expected_output.size()/2 + l);
            expected_output[ind] = init_value;
          }
        }
      }
    }
    verifier.verifyForward(input, expected_output);
    verifier.verifyBackward(expected_output, input);
  }

  // TODO: Add tests for more complex padding options
}

// TODO: neg_Pad
