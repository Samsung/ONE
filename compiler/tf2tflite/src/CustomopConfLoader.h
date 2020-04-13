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

#ifndef __CUSTOMOP_CONF_LOADER_H__
#define __CUSTOMOP_CONF_LOADER_H__

#include <moco/tf/Frontend.h>

#include <string>

namespace tf2tflite
{

/// @brief Loads customop.conf into ModelSignature
void load_customop_conf(const std::string &path, moco::ModelSignature &sig);

} // namespace tf2tflite

#endif // __CUSTOMOP_CONF_LOADER_H__
