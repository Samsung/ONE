/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include "ErrorApproximator.h"
#include "TestHelper.h"

#include <luci/IR/CircleNodeDecl.h>

#include <cmath>

namespace
{

inline uint32_t cal_offset(uint32_t shape[4], uint32_t *indices)
{
  return indices[0] * shape[1] * shape[2] * shape[3] + indices[1] * shape[2] * shape[3] +
         indices[2] * shape[3] + indices[3];
}

class NConvGraph final : public SimpleGraph
{
protected:
  void initInput(loco::Node *input) override
  {
    auto ci_input = loco::must_cast<luci::CircleNode *>(input);
    ci_input->shape_status(luci::ShapeStatus::VALID);
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->min.assign(_channel_size, _a_min);
    qparam->max.assign(_channel_size, _a_max);
    ci_input->quantparam(std::move(qparam));
  }

  loco::Node *insertGraphBody(loco::Node *input) override
  {
    _filter = _g->nodes()->create<luci::CircleConst>();
    _filter->dtype(loco::DataType::FLOAT32);
    _filter->shape({_channel_size, _f_w, _f_h, _channel_size});
    _filter->shape_status(luci::ShapeStatus::VALID);
    _filter->name("conv_filter");
    uint32_t indices[4] = {
      0,
    };

    uint32_t w_shape[4] = {_filter->dim(0).value(), _filter->dim(1).value(),
                           _filter->dim(2).value(), _filter->dim(3).value()};

    _filter->size<loco::DataType::FLOAT32>(w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]);

    for (indices[0] = 0; indices[0] < w_shape[0]; ++indices[0])
    {
      for (indices[1] = 0; indices[1] < w_shape[1]; ++indices[1])
      {
        for (indices[2] = 0; indices[2] < w_shape[2]; ++indices[2])
        {
          for (indices[3] = 0; indices[3] < w_shape[3]; ++indices[3])
          {
            uint32_t offset = cal_offset(w_shape, indices);
            _filter->at<loco::DataType::FLOAT32>(offset) = (offset % 2 == 0) ? _w_max : _w_min;
          }
        }
      }
    }

    _bias = _g->nodes()->create<luci::CircleConst>();
    _bias->dtype(loco::DataType::FLOAT32);
    _bias->shape({_channel_size});
    _bias->name("conv_bias");

    _conv = _g->nodes()->create<luci::CircleConv2D>();
    _conv->padding(luci::Padding::SAME);
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, _width, _height, _channel_size});
    _conv->shape_status(luci::ShapeStatus::VALID);
    _conv->name("conv");
    _conv->filter(_filter);
    _conv->bias(_bias);
    _conv->input(input);

    return _conv;
  }

public:
  luci::CircleConv2D *_conv = nullptr;
  luci::CircleConst *_filter = nullptr;
  luci::CircleConst *_bias = nullptr;
  uint32_t _f_w = 1;
  uint32_t _f_h = 1;
  float _w_min = -1.f;
  float _w_max = 1.f;
  float _a_min = -1.f;
  float _a_max = 1.f;
};

} // namespace

TEST(CircleMPQSolverErrorApproximatorTest, verifyResultsTest)
{
  NConvGraph g;
  g.init();

  auto value = mpqsolver::bisection::approximate(g._conv);
  float expected = ((g._w_max - g._w_min) * g._channel_size * std::max(g._a_max, g._a_min) +
                    (g._a_max - g._a_min) * g._channel_size * std::max(g._w_max, g._w_min)) *
                   g._f_h * g._f_w * g._height * g._width * g._channel_size / 1.e+6f;
  EXPECT_FLOAT_EQ(expected, value);
}

TEST(CircleMPQSolverErrorApproximatorTest, verifyResultsTest_NEG)
{
  NConvGraph g;
  g.init();

  auto value = mpqsolver::bisection::approximate(g._input);
  float expected = 0.f;
  EXPECT_FLOAT_EQ(expected, value);

  value = mpqsolver::bisection::approximate(g._output);
  expected = 0.f;
  EXPECT_FLOAT_EQ(expected, value);
}
