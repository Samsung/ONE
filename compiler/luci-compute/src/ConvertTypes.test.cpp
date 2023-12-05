/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertTypes.h"

#include <gtest/gtest.h>

TEST(ConvertTypes, tflite_shape)
{
  loco::TensorShape shape;
  shape.rank(2);
  shape.dim(0) = 1;
  shape.dim(1) = 2;

  auto tflite_shape = luci::compute::tflite_shape(shape);
  EXPECT_EQ(tflite_shape.DimensionsCount(), 2);
  EXPECT_EQ(tflite_shape.Dims(0), 1);
  EXPECT_EQ(tflite_shape.Dims(1), 2);
}

TEST(ConvertTypes, tflite_shape_NEG)
{
  loco::TensorShape shape;
  shape.rank(2);
  shape.dim(0) = 1;

  ASSERT_ANY_THROW(luci::compute::tflite_shape(shape));
}

TEST(ConvertTypes, tflite_padding)
{
  auto pts = luci::compute::PaddingType::kSame;
  ASSERT_EQ(luci::compute::tflite_padding(pts), tflite::PaddingType::kSame);
  auto ptv = luci::compute::PaddingType::kValid;
  ASSERT_EQ(luci::compute::tflite_padding(ptv), tflite::PaddingType::kValid);
}

TEST(ConvertTypes, tflite_padding_NEG)
{
  auto pt = luci::compute::PaddingType::kNone;
  ASSERT_ANY_THROW(luci::compute::tflite_padding(pt));
}

TEST(ConvertTypes, tflite_weights_format)
{
  auto fwf = luci::compute::FullyConnectedWeightsFormat::kDefault;
  ASSERT_EQ(luci::compute::tflite_weights_format(fwf),
            tflite::FullyConnectedWeightsFormat::kDefault);
}

TEST(ConvertTypes, tflite_weights_format_NEG)
{
  // force convert with invalid value as future unhandled value
  luci::compute::FullyConnectedWeightsFormat fwf =
    static_cast<luci::compute::FullyConnectedWeightsFormat>(250);
  ASSERT_ANY_THROW(luci::compute::tflite_weights_format(fwf));
}
