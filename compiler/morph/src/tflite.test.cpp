/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "morph/tflite.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

TEST(MORPH_TFLITE, as_feature_shape)
{
  auto shape = morph::tflite::as_feature_shape(tensor::Shape{1, 3, 4, 5});

  ASSERT_EQ(shape.depth(), 5);
  ASSERT_EQ(shape.height(), 3);
  ASSERT_EQ(shape.width(), 4);
}

TEST(MORPH_TFLITE, as_kernel_shape)
{
  auto shape = morph::tflite::as_kernel_shape(tensor::Shape{2, 3, 4, 5});

  ASSERT_EQ(shape.count(), 2);
  ASSERT_EQ(shape.depth(), 5);
  ASSERT_EQ(shape.height(), 3);
  ASSERT_EQ(shape.width(), 4);
}

TEST(MORPH_TFLITE, as_tensor_shape)
{
  // From feature::Shape
  {
    auto shape = morph::tflite::as_tensor_shape(feature::Shape{2, 3, 4});

    ASSERT_EQ(shape.rank(), 4);
    ASSERT_EQ(shape.dim(0), 1);
    ASSERT_EQ(shape.dim(1), 3);
    ASSERT_EQ(shape.dim(2), 4);
    ASSERT_EQ(shape.dim(3), 2);
  }

  // From kernel::Shape
  {
    auto shape = morph::tflite::as_tensor_shape(kernel::Shape{2, 3, 4, 5});

    ASSERT_EQ(shape.rank(), 4);
    ASSERT_EQ(shape.dim(0), 2);
    ASSERT_EQ(shape.dim(1), 4);
    ASSERT_EQ(shape.dim(2), 5);
    ASSERT_EQ(shape.dim(3), 3);
  }
}
