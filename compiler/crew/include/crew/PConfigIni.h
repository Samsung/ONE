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

#ifndef __CREW_PCONFIG_INI_H__
#define __CREW_PCONFIG_INI_H__

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace crew
{

using KeyValues = std::unordered_map<std::string, std::string>;

struct Section
{
  std::string name;
  KeyValues items;
};

using Sections = std::vector<Section>;

/**
 * @brief Reads Config INI from null terminated string and return Sections
 */
Sections read_ini(const char *data, size_t length);
/**
 * @brief Reads Config INI from file and return Sections
 */
Sections read_ini(const std::string &path);

/**
 * @brief Write Config INI with Sections to ostream
 */
void write_ini(std::ostream &os, const Sections &sections);
/**
 * @brief Write Config INI with Sections to file, throw if failed
 */
void write_ini(const std::string &path, const Sections &sections);

/**
 * @brief Find a section with name, empty section if not found
 */
Section find(const Sections &sections, const std::string &name);

/**
 * @brief Find a key-value pair from key and return value, empty string if not found
 */
std::string find(const Section &section, const std::string &key);

} // namespace crew

#endif // __CREW_PCONFIG_INI_H__
