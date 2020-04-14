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

extern const char *MODEL_ONE_OP_IN_TFLITE;

/**
 * @brief A helper class to find models for testing
 */
class ModelPath
{
public:
  static ModelPath &get();

  /**
   * @brief Get the Absolute of the model to find
   *
   * @param model_dir A relative model directory
   * @return std::string The absolute path of model directory
   */
  std::string getModelAbsolutePath(const char *model_dir);
  /**
   * @brief Save the current executable's directory based on argv[0] and CWD
   *
   * @param argv0 0th command line argument of the current process
   */
  void init(const char *argv0);

private:
  ModelPath() = default;

private:
  std::string _base_path;
};

#endif // __NNFW_API_TEST_MODEL_PATH_H__
