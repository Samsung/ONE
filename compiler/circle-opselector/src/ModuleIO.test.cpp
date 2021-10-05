/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TestHelper.h"

#include <gtest/gtest.h>

TEST(ModuleIOTest, Export)
{
  std::string input_path = "../common-artifacts/Part_Sqrt_Rsqrt_002.circle";
  std::string output_path = "./test.out.circle";

  auto module = opselector::getModule(input_path);

  ASSERT_EQ(true, opselector::exportModule(module.get(), output_path));
  ASSERT_EQ(false, opselector::exportModule(nullptr, output_path));
}
