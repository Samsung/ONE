/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CREW_PCONFIG_H__
#define __CREW_PCONFIG_H__

#include <iostream>
#include <string>
#include <vector>

namespace crew
{

struct Part
{
  std::string model_file;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

using Parts = std::vector<Part>;
using Source = Part;

struct PConfig
{
  Source source;
  Parts parts;
};

/**
 * @brief Read config as ini file, return false if failed
 */
bool read_ini(const std::string &path, PConfig &config);

} // namespace crew

#endif // __CREW_PCONFIG_H__
