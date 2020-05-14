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

#ifndef __RAW_MODEL_H__
#define __RAW_MODEL_H__

#include <mio/circle/schema_generated.h>

namespace circlechef
{

struct RawModel
{
  virtual ~RawModel() = default;

  virtual const ::circle::Model *model(void) const = 0;
};

/**
 * @brief Load circle model (as a RawModel) from a given path
 *
 * @note May return a nullptr
 */
std::unique_ptr<RawModel> load_circle(const std::string &path);

} // namespace circlechef

#endif // __RAW_MODEL_H__
