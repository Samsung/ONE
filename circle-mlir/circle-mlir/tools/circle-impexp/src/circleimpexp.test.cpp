/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <string>

// delcare methods of circleimpexp.cpp
int circleImport(const std::string &sourcefile, std::string &model_string);

#include <gtest/gtest.h>

TEST(CircleImportTest, NonExistFile_NEG)
{
  std::string invalid_filename = "/no_such_folder/no_such_file_in_storage";
  std::string model_string = "";

  auto result = circleImport(invalid_filename, model_string);
  ASSERT_NE(0, result);
}
