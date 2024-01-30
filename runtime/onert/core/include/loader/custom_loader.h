/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_LOADER_CUSTOM_LOADER_H__
#define __ONERT_LOADER_CUSTOM_LOADER_H__

#include "ir/Graph.h"

#include <memory>

namespace onert
{
namespace custom_loader
{

class ICustomLoader
{
public:
  virtual ~ICustomLoader() = default;

public:
  /**
   * @brief     Load model from file
   * @param[in] file_path File path to load model from
   * @return    Loaded model
   */
  virtual std::unique_ptr<ir::Model> loadFromFile(const std::string &file_path) = 0;
};

/**
 * @brief     Create custom loader and load model from file
 * @param[in] filename  File path to load model from
 * @param[in] type      Type of custom loader to create
 * @return    Loaded model.
 *
 * @note  Throw exception if failed to load model
 */
std::unique_ptr<ir::Model> loadModel(const std::string &filename, const std::string &type);
} // namespace custom_loader
} // namespace onert

#endif // __ONERT_LOADER_CUSTOM_LOADER_H__
