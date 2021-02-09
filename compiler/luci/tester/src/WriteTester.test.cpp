/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

// From WriteTester.cpp
int entry(int argc, char **argv);

TEST(WriteTesterTest, invalid_argc_NEG)
{
  char argv_1[20];
  strcpy(argv_1, "WriteTesterTest");

  int argc = 1;
  char *argv[] = {argv_1};

  ASSERT_NE(0, entry(argc, argv));
}

TEST(WriteTesterTest, invalid_file_NEG)
{
  char argv_1[20], argv_2[20], argv_3[20];
  strcpy(argv_1, "WriteTesterTest");
  strcpy(argv_2, "not_a_file");
  strcpy(argv_3, "not_a_file");

  int argc = 3;
  char *argv[] = {argv_1, argv_2, argv_3};

  EXPECT_THROW(entry(argc, argv), std::runtime_error);
}
