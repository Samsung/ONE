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

#include <iostream>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>
#include "NNPackages.h"

int main(int argc, char **argv)
{
  NNPackages::get().init(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);

  try
  {
    NNPackages::get().checkAll();
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "[WARNING] Test models are not loaded, so some tests will fail" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  return RUN_ALL_TESTS();
}
