/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <array>
#include <gtest/gtest.h>
#include <initializer_list>
#include <sstream>
#include <vector>
#include <stdexcept>

using namespace nnfw::cker;

TEST(CKer_Utils, Shape)
{
  // SmallShape
  {
    // Create a small shape using an initializer list.
    Shape s({2, 3, 4});
    EXPECT_EQ(s.DimensionsCount(), 3);
    EXPECT_EQ(s.Dims(0), 2);
    EXPECT_EQ(s.Dims(1), 3);
    EXPECT_EQ(s.Dims(2), 4);
    EXPECT_EQ(s.FlatSize(), 2 * 3 * 4);
  }

  // LargeShape
  {
    // Create a shape with 6 dimensions, which is larger than kMaxSmallSize.
    int dims[6] = {1, 2, 3, 4, 5, 6};
    Shape s(6, dims);
    EXPECT_EQ(s.DimensionsCount(), 6);
    for (int i = 0; i < 6; ++i)
    {
      EXPECT_EQ(s.Dims(i), dims[i]);
    }
  }

  // InitializerListConstructor
  {
    // Using initializer list.
    Shape s{2, 3, 4, 5};
    EXPECT_EQ(s.DimensionsCount(), 4);
    EXPECT_EQ(s.Dims(0), 2);
    EXPECT_EQ(s.Dims(1), 3);
    EXPECT_EQ(s.Dims(2), 4);
    EXPECT_EQ(s.Dims(3), 5);
  }

  // CopyConstructor
  {
    Shape s1{2, 3, 4};
    Shape s2(s1);
    EXPECT_EQ(s1, s2);
  }

  // FlatSize
  {
    Shape s{2, 3, 4};
    EXPECT_EQ(s.FlatSize(), 2 * 3 * 4);
  }

  // DimsDataAccess
  {
    Shape s{2, 3, 4};
    int32_t *data = s.DimsData();
    // Set the dims via the raw pointer and check with Dims().
    data[0] = 5;
    data[1] = 6;
    data[2] = 7;
    EXPECT_EQ(s.Dims(0), 5);
    EXPECT_EQ(s.Dims(1), 6);
    EXPECT_EQ(s.Dims(2), 7);
  }

  // ExtendedShape
  {
    // For a 2D shape [2, 3] extended to 5 dimensions, ExtendedShape pads the shape with 1.
    Shape s({2, 3});
    Shape extended = Shape::ExtendedShape(5, s);
    EXPECT_EQ(extended.DimensionsCount(), 5);
    EXPECT_EQ(extended.Dims(0), 1);
    EXPECT_EQ(extended.Dims(1), 1);
    EXPECT_EQ(extended.Dims(2), 1);
    EXPECT_EQ(extended.Dims(3), 2);
    EXPECT_EQ(extended.Dims(4), 3);
  }

  // MatchingDim (single)
  {
    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    int dim = MatchingDim(s1, 1, s2, 1);
    EXPECT_EQ(dim, 3);
  }

  // MatchingDim (multiple)
  {
    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    Shape s3{2, 3, 4};
    int dim = MatchingDim(s1, 0, s2, 0, s3, 0);
    EXPECT_EQ(dim, 2);
  }

  // FlatSizeSkipDim
  {
    // For shape [2, 3, 4, 5], skipping dim 2 should compute 2*3*5 = 30.
    int dims[4] = {2, 3, 4, 5};
    Shape s(4, dims);
    int flat_size = FlatSizeSkipDim(s, 2);
    EXPECT_EQ(flat_size, 2 * 3 * 5);
  }

  // CheckMatchingTrue
  {
    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    Shape s3{2, 3, 4};
    EXPECT_TRUE(checkMatching(s1, s2, s3));
  }

  // MatchingFlatSize
  {
    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    int flat_size = MatchingFlatSize(s1, s2);
    EXPECT_EQ(flat_size, s1.FlatSize());
  }

  // MatchingFlatSizeSkipDim
  {
    Shape s1{2, 3, 4, 5};
    int fs1 = MatchingFlatSizeSkipDim(s1, 2, s1);
    int fs2 = MatchingFlatSizeSkipDim(s1, 2, s1, s1);
    EXPECT_EQ(fs1, FlatSizeSkipDim(s1, 2));
    EXPECT_EQ(fs2, FlatSizeSkipDim(s1, 2));
  }

  // MatchingElementsSize
  {
    Shape s1{2, 3, 4};
    Shape s2{2, 3, 4};
    Shape s3{2, 3, 4};
    int size = MatchingElementsSize(s1, s2, s3);
    EXPECT_EQ(size, s1.FlatSize());
  }

  // GetShapeHelper
  {
    std::vector<int32_t> dims = {2, 3, 4};
    Shape s = GetShape(dims);
    EXPECT_EQ(s.DimensionsCount(), 3);
    EXPECT_EQ(s.Dims(0), 2);
    EXPECT_EQ(s.Dims(1), 3);
    EXPECT_EQ(s.Dims(2), 4);
  }

  // OffsetFunction
  {
    // For a 4D shape [2, 3, 4, 5].
    int dims[4] = {2, 3, 4, 5};
    Shape s(4, dims);
    int off = Offset(s, 1, 2, 3, 4);
    // Expected offset: ((1*3+2)*4+3)*5+4 = 119.
    EXPECT_EQ(off, 119);
  }

  // OffsetFunctionArrayIndex
  {
    int dims[4] = {2, 3, 4, 5};
    Shape s(4, dims);
    int indices[4] = {1, 2, 3, 4};
    int off = Offset(s, indices);
    EXPECT_EQ(off, 119);
  }
}

// neg_CheckMatching: Test that checkMatching returns false when shapes do not match.
TEST(CKer_Utils, neg_ShapeCheckMatching)
{
  Shape s1({2, 3, 4});
  Shape s2({2, 3, 5}); // Last dimension differs.
  EXPECT_FALSE(checkMatching(s1, s2));
}
