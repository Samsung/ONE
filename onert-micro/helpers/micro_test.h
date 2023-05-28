/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ONERT_MICRO_HELPERS_TEST_H
#define ONERT_MICRO_HELPERS_TEST_H
#include <mbed.h>
#include <iostream>
#include <limits>
#include <type_traits>

namespace micro_test
{
extern int tests_passed;
extern int tests_failed;
extern bool is_test_complete;
extern bool did_test_fail;
} // namespace micro_test

#define ONERT_MICRO_TESTS_BEGIN   \
  namespace micro_test            \
  {                               \
  int tests_passed;               \
  int tests_failed;               \
  bool is_test_complete;          \
  bool did_test_fail;             \
  }                               \
                                  \
  int main(int argc, char **argv) \
  {                               \
    micro_test::tests_passed = 0; \
    micro_test::tests_failed = 0;

#define ONERT_MICRO_TESTS_END                                    \
  printf("%d/%d tests passed\n", micro_test::tests_passed,       \
         (micro_test::tests_failed + micro_test::tests_passed)); \
  if (micro_test::tests_failed == 0)                             \
  {                                                              \
    printf("~~~ALL TESTS PASSED~~~\n");                          \
    return 0;                                                    \
  }                                                              \
  else                                                           \
  {                                                              \
    printf("~~~SOME TESTS FAILED~~~\n");                         \
    return -1;                                                   \
  }                                                              \
  }

#define ONERT_MICRO_TEST(name)                                                  \
  printf("Testing " #name "\n");                                                \
  for (micro_test::is_test_complete = false, micro_test::did_test_fail = false; \
       !micro_test::is_test_complete; micro_test::is_test_complete = true,      \
      micro_test::tests_passed += (micro_test::did_test_fail) ? 0 : 1,          \
      micro_test::tests_failed += (micro_test::did_test_fail) ? 1 : 0)

#define ONERT_MICRO_EXPECT(x)                              \
  do                                                       \
  {                                                        \
    if (!(x))                                              \
    {                                                      \
      printf(#x " failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                    \
    }                                                      \
  } while (false)

#define ONERT_MICRO_EXPECT_EQ(x, y)                                              \
  do                                                                             \
  {                                                                              \
    auto vx = x;                                                                 \
    auto vy = y;                                                                 \
    bool isFloatingX = (std::is_floating_point<decltype(vx)>::value);            \
    bool isFloatingY = (std::is_floating_point<decltype(vy)>::value);            \
    if (isFloatingX && isFloatingY)                                              \
    {                                                                            \
      auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));                \
      if (delta > std::numeric_limits<decltype(delta)>::epsilon())               \
      {                                                                          \
        printf(#x " == " #y " failed at %s:%d (%f vs %f)\n", __FILE__, __LINE__, \
               static_cast<double>(vx), static_cast<double>(vy));                \
        micro_test::did_test_fail = true;                                        \
      }                                                                          \
    }                                                                            \
    else if ((vx) != (vy))                                                       \
    {                                                                            \
      printf(#x " == " #y " failed at %s:%d (%d vs %d)\n", __FILE__, __LINE__,   \
             static_cast<int>(vx), static_cast<int>(vy));                        \
      if (isFloatingX || isFloatingY)                                            \
      {                                                                          \
        printf("-----------WARNING-----------\n");                               \
        printf("Only one of the values is floating point value.\n");             \
      }                                                                          \
      micro_test::did_test_fail = true;                                          \
    }                                                                            \
  } while (false)

#define ONERT_MICRO_EXPECT_NE(x, y)                                   \
  do                                                                  \
  {                                                                   \
    auto vx = x;                                                      \
    auto vy = y;                                                      \
    bool isFloatingX = (std::is_floating_point<decltype(vx)>::value); \
    bool isFloatingY = (std::is_floating_point<decltype(vy)>::value); \
    if (isFloatingX && isFloatingY)                                   \
    {                                                                 \
      auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));     \
      if (delta <= std::numeric_limits<decltype(delta)>::epsilon())   \
      {                                                               \
        printf(#x " != " #y " failed at %s:%d", __FILE__, __LINE__);  \
        micro_test::did_test_fail = true;                             \
      }                                                               \
    }                                                                 \
    else if ((vx) == (vy))                                            \
    {                                                                 \
      printf(#x " != " #y " failed at %s:%d", __FILE__, __LINE__);    \
      if (isFloatingX || isFloatingY)                                 \
      {                                                               \
        printf("-----------WARNING-----------\n");                    \
        printf("Only one of the values is floating point value.\n");  \
      }                                                               \
      micro_test::did_test_fail = true;                               \
    }                                                                 \
  } while (false)

#define ONERT_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(arr1, idx1, arr2, idx2, epsilon)                    \
  do                                                                                              \
  {                                                                                               \
    auto delta = ((arr1)[(idx1)] > (arr2)[(idx2)]) ? ((arr1)[(idx1)] - (arr2)[(idx2)])            \
                                                   : ((arr2)[(idx2)] - (arr1)[(idx1)]);           \
    if (delta > epsilon)                                                                          \
    {                                                                                             \
      printf(#arr1 "[%d] (%f) near " #arr2 "[%d] (%f) failed at %s:%d\n", static_cast<int>(idx1), \
             static_cast<float>((arr1)[(idx1)]), static_cast<int>(idx2),                          \
             static_cast<float>((arr2)[(idx2)]), __FILE__, __LINE__);                             \
      micro_test::did_test_fail = true;                                                           \
    }                                                                                             \
  } while (false)

#define ONERT_MICRO_EXPECT_NEAR(x, y, epsilon)                                       \
  do                                                                                 \
  {                                                                                  \
    auto vx = (x);                                                                   \
    auto vy = (y);                                                                   \
    auto delta = ((vx) > (vy)) ? ((vx) - (vy)) : ((vy) - (vx));                      \
    if (vx != vy && delta > epsilon)                                                 \
    {                                                                                \
      printf(#x " (%f) near " #y " (%f) failed at %s:%d\n", static_cast<double>(vx), \
             static_cast<double>(vy), __FILE__, __LINE__);                           \
      micro_test::did_test_fail = true;                                              \
    }                                                                                \
  } while (false)

#define ONERT_MICRO_EXPECT_GT(x, y)                                 \
  do                                                                \
  {                                                                 \
    if ((x) <= (y))                                                 \
    {                                                               \
      printf(#x " > " #y " failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                             \
    }                                                               \
  } while (false)

#define ONERT_MICRO_EXPECT_LT(x, y)                                 \
  do                                                                \
  {                                                                 \
    if ((x) >= (y))                                                 \
    {                                                               \
      printf(#x " < " #y " failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                             \
    }                                                               \
  } while (false)

#define ONERT_MICRO_EXPECT_GE(x, y)                                  \
  do                                                                 \
  {                                                                  \
    if ((x) < (y))                                                   \
    {                                                                \
      printf(#x " >= " #y " failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                              \
    }                                                                \
  } while (false)

#define ONERT_MICRO_EXPECT_LE(x, y)                                  \
  do                                                                 \
  {                                                                  \
    if ((x) > (y))                                                   \
    {                                                                \
      printf(#x " <= " #y " failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                              \
    }                                                                \
  } while (false)

#define ONERT_MICRO_EXPECT_TRUE(x)                                      \
  do                                                                    \
  {                                                                     \
    if (!(x))                                                           \
    {                                                                   \
      printf(#x " was not true failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                                 \
    }                                                                   \
  } while (false)

#define ONERT_MICRO_EXPECT_FALSE(x)                                      \
  do                                                                     \
  {                                                                      \
    if (x)                                                               \
    {                                                                    \
      printf(#x " was not false failed at %s:%d\n", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                                  \
    }                                                                    \
  } while (false)

#define ONERT_MICRO_FAIL(msg)                      \
  do                                               \
  {                                                \
    printf("FAIL: %s\n", msg, __FILE__, __LINE__); \
    micro_test::did_test_fail = true;              \
  } while (false)

#define ONERT_MICRO_EXPECT_STRING_EQ(string1, string2)                               \
  do                                                                                 \
  {                                                                                  \
    for (int i = 0; string1[i] != '\0' && string2[i] != '\0'; i++)                   \
    {                                                                                \
      if (string1[i] != string2[i])                                                  \
      {                                                                              \
        printf("FAIL: %s did not match %s\n", string1, string2, __FILE__, __LINE__); \
        micro_test::did_test_fail = true;                                            \
      }                                                                              \
    }                                                                                \
  } while (false)

#endif // ONERT_MICRO_HELPERS_TEST_H
