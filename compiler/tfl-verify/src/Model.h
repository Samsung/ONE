/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MODEL_H__
#define __MODEL_H__

#include <memory>
#include <string>

struct ModelData
{
  virtual ~ModelData() = default;

  virtual const void *data(void) const = 0;
  virtual const size_t size(void) const = 0;
};

/**
 * @brief Load TF Lite model (as a raw data) from a given path
 *
 * @note May return a nullptr
 */
std::unique_ptr<ModelData> load_modeldata(const std::string &path);

#endif // __MODEL_H__
