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

#include "NNPackages.h"

#include <unistd.h>
#include <libgen.h>
#include <string.h>

// NOTE Must match `enum TestPackages`
const char *TEST_PACKAGE_NAMES[] = {"nonexisting_package", "add"};

NNPackages &NNPackages::get()
{
  static NNPackages instance;
  return instance;
}

void NNPackages::init(const char *argv0)
{
  char raw_dir[1024];
  char cwd[1024];
  strncpy(raw_dir, argv0, sizeof(raw_dir) - 1);
  dirname(raw_dir);
  if (raw_dir[0] == '/')
  {
    // If it is an absolute path, just use it
    _base_path = raw_dir;
  }
  else
  {
    // If it is a relative path, prepend CWD
    getcwd(cwd, sizeof(cwd));
    _base_path = cwd;
    _base_path += "/";
    _base_path += raw_dir;
  }
}

std::string NNPackages::getModelAbsolutePath(int package_no)
{
  const char *model_dir = TEST_PACKAGE_NAMES[package_no];
  // Model dir is nested
  return _base_path + "/nnfw_api_gtest_models/" + model_dir + "/" + model_dir;
}
