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
#include <dirent.h>
#include <assert.h>
#include <stdexcept>

// NOTE Must match `enum TestPackages`
const char *TEST_PACKAGE_NAMES[] = {
  // for validation test
  "add",
  "add_no_manifest",
  "add_invalid_manifest",

  // for dynamic tensor test
  "while_dynamic",
  "if_dynamic",
};

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
  char *dir_path = dirname(raw_dir);
  if (dir_path[0] == '/')
  {
    // If it is an absolute path, just use it
    _base_path = dir_path;
  }
  else
  {
    // If it is a relative path, prepend CWD
    getcwd(cwd, sizeof(cwd));
    _base_path = cwd;
    _base_path += "/";
    _base_path += dir_path;
  }
}

void NNPackages::checkAll()
{
  assert(!_base_path.empty());

  for (int i = 0; i < NNPackages::COUNT; i++)
  {
    std::string package_name = TEST_PACKAGE_NAMES[i];
    std::string path = getModelAbsolutePath(i);

    DIR *dir = opendir(path.c_str());
    if (!dir)
    {
      std::string msg = "missing nnpackage: " + package_name + ", path: " + path +
                        "\nPlease run \'[install_dir]/test/onert-test prepare-model\' to "
                        "download nnpackage";
      throw std::runtime_error{msg};
    }
    closedir(dir);
  }
}

std::string NNPackages::getModelAbsolutePath(int package_no)
{
  if (package_no < 0 || package_no >= NNPackages::COUNT)
  {
    throw std::runtime_error{"Invalid package_no: " + std::to_string(package_no)};
  }

  const char *package_dir = TEST_PACKAGE_NAMES[package_no];
  return getModelAbsolutePath(package_dir);
}

std::string NNPackages::getModelAbsolutePath(const char *package_name)
{
  return _base_path + "/nnfw_api_gtest_models/" + package_name + "/" + package_name;
}

std::string NNPackages::getModelAbsoluteFilePath(const char *package_name)
{
  return _base_path + "/nnfw_api_gtest_models/" + package_name + "/" + package_name + ".tflite";
}
