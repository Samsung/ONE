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

#ifndef __NNFW_API_TEST_MODEL_PATH_H__
#define __NNFW_API_TEST_MODEL_PATH_H__

#include <string>

/**
 * @brief A helper class to find NN Packages for testing
 *        To add a nnpackage for your test, please do the followings:
 *          0. Prerequisite: the actual file must be uploaded on the server
 *                           Add `config.sh` file to `tests/scripts/models/nnfw_api_gtest`
 *          1. Append an enum value to @c NNPackages::TestPackages
 *          2. Append a string literal to @c TEST_PACKAGE_NAMES in the source file
 */
class NNPackages
{
public:
  /**
   * @brief Serial numbers for test packages. The numbers are mapped with package names.
   *        This is useful for creating GTest Fixtures with variable template to perform
   *        different nn packages with no code duplication.
   */
  enum TestPackages
  {
    // for validation test
    ADD,
    ADD_NO_MANIFEST,      //< Contains "Add" model but no manifest file
    ADD_INVALID_MANIFEST, //< Contains "Add" model but the manifest file is broken JSON

    // for dynamic tensor test
    WHILE_DYNAMIC,
    IF_DYNAMIC,

    COUNT
  };

  /*
   * @brief Singleton object getter
   *
   * @return NNPackages& The singleton object
   */
  static NNPackages &get();

  /**
   * @brief Get the Absolute of the model to find
   *
   * @param package_no Model's serial number
   * @return std::string The absolute path of model directory
   */
  std::string getModelAbsolutePath(int package_no);

  /**
   * @brief Get the absolute of the model to find
   *
   * @param package_name Package name
   * @return std::string The absolute path of model directory
   */
  std::string getModelAbsolutePath(const char *package_name);

  /**
   * @brief Get the absolute of the model file to find
   *
   * @param package_name Package name
   * @return std::string The absolute path of model file
   */
  std::string getModelAbsoluteFilePath(const char *package_name);

  /**
   * @brief Save the current executable's directory based on argv[0] and CWD
   *
   * @param argv0 0th command line argument of the current process
   */
  void init(const char *argv0);

  /**
   * @brief Check all the nnpackages are installed
   *        Must be run after @c init .
   */
  void checkAll();

private:
  NNPackages() = default;

private:
  std::string _base_path;
};

#endif // __NNFW_API_TEST_MODEL_PATH_H__
