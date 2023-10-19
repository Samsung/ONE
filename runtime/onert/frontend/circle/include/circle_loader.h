/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_CIRCLE_LOADER_H__
#define __CIRCLE_CIRCLE_LOADER_H__

#include "ir/Graph.h"

#include <memory>

namespace onert
{
namespace circle_loader
{
std::unique_ptr<ir::Model> loadModel(const std::string &filename);
std::unique_ptr<ir::Model> loadModel(const std::string &filename,
                                     std::vector<std::string> metadata_names);
std::unique_ptr<ir::Model> loadModel(uint8_t *buffer, size_t size);
} // namespace circle_loader
} // namespace onert

#endif // __CIRCLE_CIRCLE_LOADER_H__
