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

#include "DeathTestMacro.h"

#include <functional>
#include <gtest/gtest.h>
#include <vector>

using namespace nnfw::cker;

TEST(ShapeTest, DefaultConstructor)
{
  // Create an empty shape.
  Shape s;
  EXPECT_EQ(s.DimensionsCount(), 0);
  // FlatSize returns 1 when there are no dimensions.
  EXPECT_EQ(s.FlatSize(), 1);
}

TEST(ShapeTest, DimensionCountConstructorAndSetGet)
{
  // Create a shape with 3 dimensions; set the values.
  Shape s(3);
  EXPECT_EQ(s.DimensionsCount(), 3);
  s.SetDim(0, 2);
  s.SetDim(1, 3);
  s.SetDim(2, 4);
  EXPECT_EQ(s.Dims(0), 2);
  EXPECT_EQ(s.Dims(1), 3);
  EXPECT_EQ(s.Dims(2), 4);
  EXPECT_EQ(s.FlatSize(), 2 * 3 * 4);
}

TEST(ShapeTest, FillConstructor)
{
  // Constructor that fills all dimensions with the given value.
  Shape s(4, 5);
  EXPECT_EQ(s.DimensionsCount(), 4);
  for (int i = 0; i < 4; ++i)
  {
    EXPECT_EQ(s.Dims(i), 5);
  }
  EXPECT_EQ(s.FlatSize(), 5 * 5 * 5 * 5);
}

TEST(ShapeTest, ArrayConstructor)
{
  // Create a shape from a C-array.
  int32_t dims[] = {3, 4, 5};
  Shape s(3, dims);
  EXPECT_EQ(s.DimensionsCount(), 3);
  EXPECT_EQ(s.Dims(0), 3);
  EXPECT_EQ(s.Dims(1), 4);
  EXPECT_EQ(s.Dims(2), 5);
  EXPECT_EQ(s.FlatSize(), 3 * 4 * 5);
}

TEST(ShapeTest, InitializerListConstructor)
{
  // Create a shape with an initializer list.
  Shape s{1, 2, 3, 4};
  EXPECT_EQ(s.DimensionsCount(), 4);
  EXPECT_EQ(s.Dims(0), 1);
  EXPECT_EQ(s.Dims(1), 2);
  EXPECT_EQ(s.Dims(2), 3);
  EXPECT_EQ(s.Dims(3), 4);
  EXPECT_EQ(s.FlatSize(), 1 * 2 * 3 * 4);
}

TEST(ShapeTest, CopyAndMoveConstructors)
{
  // Test that copying a shape produces an equivalent shape.
  Shape s{2, 3, 4};
  Shape t(s);
  EXPECT_EQ(t.DimensionsCount(), s.DimensionsCount());
  for (int i = 0; i < s.DimensionsCount(); ++i)
    EXPECT_EQ(t.Dims(i), s.Dims(i));

  // Test move constructor.
  Shape u(std::move(t));
  EXPECT_EQ(u.DimensionsCount(), s.DimensionsCount());
  for (int i = 0; i < s.DimensionsCount(); ++i)
    EXPECT_EQ(u.Dims(i), s.Dims(i));
}

TEST(ShapeTest, ResizePreservesData)
{
  // Create a shape and then resize it.
  Shape s{2, 3};
  s.SetDim(0, 2);
  s.SetDim(1, 3);
  s.Resize(4);
  EXPECT_EQ(s.DimensionsCount(), 4);
  EXPECT_EQ(s.Dims(0), 2);
  EXPECT_EQ(s.Dims(1), 3);
  // New dimensions should be default-initialized to 0.
  EXPECT_EQ(s.Dims(2), 0);
  EXPECT_EQ(s.Dims(3), 0);
}

TEST(ShapeTest, ReplaceWithAndBuildFrom)
{
  // Test replacing the shape with new data.
  Shape s{2, 3, 4};
  int32_t new_dims[] = {5, 6, 7, 8};
  s.ReplaceWith(4, new_dims);
  EXPECT_EQ(s.DimensionsCount(), 4);
  EXPECT_EQ(s.Dims(0), 5);
  EXPECT_EQ(s.Dims(1), 6);
  EXPECT_EQ(s.Dims(2), 7);
  EXPECT_EQ(s.Dims(3), 8);

  // Test building a shape from an initializer list.
  s.BuildFrom({9, 10});
  EXPECT_EQ(s.DimensionsCount(), 2);
  EXPECT_EQ(s.Dims(0), 9);
  EXPECT_EQ(s.Dims(1), 10);
}

TEST(ShapeTest, ExtendedShape)
{
  // Test creating an extended shape with padding.
  // The ExtendedShape function pads the shape so that the pad portion comes
  // first, followed by the original dimensions.
  Shape base{3, 4};
  // Extend to 6 dimensions; with a pad value of 1.
  Shape extended = Shape::ExtendedShape(6, base);
  EXPECT_EQ(extended.DimensionsCount(), 6);
  // For a base shape with 2 dims, size_increase = 4.
  for (int i = 0; i < 4; i++)
  {
    EXPECT_EQ(extended.Dims(i), 1);
  }
  EXPECT_EQ(extended.Dims(4), 3);
  EXPECT_EQ(extended.Dims(5), 4);
}

TEST(ShapeTest, FlatSizeCalculation)
{
  // Test that FlatSize returns the product of the dimensions.
  Shape s{2, 3, 4};
  EXPECT_EQ(s.FlatSize(), 2 * 3 * 4);
}

TEST(ShapeTest, GetDimsData)
{
  // Test the DimsData() accessor.
  Shape s(3);
  s.SetDim(0, 7);
  s.SetDim(1, 8);
  s.SetDim(2, 9);
  const int32_t *data = s.DimsData();
  EXPECT_EQ(data[0], 7);
  EXPECT_EQ(data[1], 8);
  EXPECT_EQ(data[2], 9);
}

TEST(ShapeTest, UtilityFunctions)
{
  // Test some of the utility functions.
  Shape s{2, 3, 4};
  // MatchingDim should return the matching value at a given index.
  int m = MatchingDim(s, 1, s, 1);
  EXPECT_EQ(m, 3);
  // MatchingFlatSize should return the shape's flat size.
  int fs = MatchingFlatSize(s, s);
  EXPECT_EQ(fs, 2 * 3 * 4);
  // MatchingFlatSizeSkipDim: skipping dimension 1, expect 2 * 4.
  int fs_skip = MatchingFlatSizeSkipDim(s, 1, s);
  EXPECT_EQ(fs_skip, 2 * 4);
}

// Negative tests using assertions and EXPECT_DEATH

// neg_CheckMatching: Test that checkMatching returns false when shapes do not match.
TEST(CKer_Utils, neg_ShapeCheckMatching)
{
  Shape s1({2, 3, 4});
  Shape s2({2, 3, 5}); // Last dimension differs.
  EXPECT_FALSE(checkMatching(s1, s2));
}

// Test that accessing a dimension out of range triggers an assertion.
TEST(ShapeTest, neg_DimsAccessOutOfRange)
{
  Shape s{1, 2, 3};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ s.Dims(3); }, ".*");
}

// Test that setting a dimension out of range triggers an assertion.
TEST(ShapeTest, neg_SetDimOutOfRange)
{
  Shape s{1, 2, 3};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ s.SetDim(3, 10); }, ".*");
}

// Test that accessing a dimension using a negative index triggers an assertion.
TEST(ShapeTest, neg_DimsAccessNegativeIndex)
{
  Shape s{1, 2, 3};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ s.Dims(-1); }, ".*");
}

// Test that setting a dimension using a negative index triggers an assertion.
TEST(ShapeTest, neg_SetDimNegativeIndex)
{
  Shape s{1, 2, 3};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ s.SetDim(-1, 10); }, ".*");
}

// Test that creating an extended shape with new_shape_size less than the base's dimension count
// triggers an assertion.
TEST(ShapeTest, neg_ExtendedShapeInvalid)
{
  Shape base{4, 5, 6};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ Shape::ExtendedShape(2, base); }, ".*");
}

// Test that mismatched dimensions for MatchingDim triggers an assertion.
TEST(ShapeTest, neg_MatchingDimMismatch)
{
  Shape s1{2, 3, 4};
  Shape s2{2, 3, 5};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ MatchingDim(s1, 2, s2, 2); }, ".*");
}

// Test that mismatched shapes for MatchingFlatSize triggers an assertion.
TEST(ShapeTest, neg_MatchingFlatSizeMismatch)
{
  Shape s1{2, 3, 4};
  Shape s2{2, 3, 5};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ MatchingFlatSize(s1, s2); }, ".*");
}

// Test that using an out-of-bound skip dimension for MatchingFlatSizeSkipDim triggers an
// assertion.
TEST(ShapeTest, neg_MatchingFlatSizeSkipDimOutOfBound)
{
  Shape s{2, 3, 4};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ MatchingFlatSizeSkipDim(s, 3, s); }, ".*");
}

// Test that calling ReplaceWith with an invalid (null) pointer triggers an assertion.
TEST(ShapeTest, neg_ReplaceWithInvalidDimensions)
{
  Shape s{3};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY({ s.ReplaceWith(3, nullptr); }, ".*");
}

// Test that constructing a Shape with a negative number of dimensions triggers an assertion.
TEST(ShapeTest, neg_ConstructorWithNegativeDimensions)
{
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      Shape s(-3);
      s.DimensionsCount();
    },
    ".*");
}

// Test that calling Offset on a 4D shape with a negative index triggers an assertion.
TEST(ShapeTest, neg_Offset4DNegativeIndex)
{
  Shape s{2, 3, 4, 5};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      int offset = Offset(s, -1, 1, 1, 1);
      (void)offset;
    },
    ".*");
}

// Test that calling Offset on a 5D shape with a negative index triggers an assertion.
TEST(ShapeTest, neg_Offset5DNegativeIndex)
{
  Shape s{2, 3, 4, 5, 6};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      int offset = Offset(s, 1, 1, -1, 1, 1);
      (void)offset;
    },
    ".*");
}

// Test that calling Offset on a 6D shape with an out-of-range index triggers an assertion.
TEST(ShapeTest, neg_Offset6DIndexOutOfRange)
{
  Shape s{2, 3, 4, 5, 6, 7};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      int offset = Offset(s, 1, 1, 1, 1, 1, 10);
      (void)offset;
    },
    ".*");
}

// Test that calling Offset on a 6D shape with a negative index triggers an assertion.
TEST(ShapeTest, neg_Offset6DNegativeIndex)
{
  Shape s{2, 3, 4, 5, 6, 7};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      int offset = Offset(s, 1, -2, 1, 1, 1, 1);
      (void)offset;
    },
    ".*");
}

// Test that calling MatchingFlatSizeSkipDim with a negative skip dimension triggers an assertion.
TEST(ShapeTest, neg_MatchingFlatSizeSkipDimNegative)
{
  Shape s{2, 3, 4};
  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      int flat = MatchingFlatSizeSkipDim(s, -1, s);
      (void)flat;
    },
    ".*");
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";

  return RUN_ALL_TESTS();
}
