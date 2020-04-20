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
#include <dirent.h>
#include <gtest/gtest.h>
#include "model_path.h"

/**
 * @brief Function to check if test model directories exist before it actually performs the test
 *
 */
void checkModels()
{
  std::string absolute_path = ModelPath::get().getModelAbsolutePath(MODEL_ADD);
  DIR *dir = opendir(absolute_path.c_str());
  if (!dir)
  {
    throw std::runtime_error{"Please install the nnpackge for testing: " + absolute_path};
  }
  closedir(dir);
}

int main(int argc, char **argv)
{
  ModelPath::get().init(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);

  try
  {
    checkModels();
  }
  catch (std::runtime_error &e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return RUN_ALL_TESTS();
}
