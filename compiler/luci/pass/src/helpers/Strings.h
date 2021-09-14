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

#ifndef __LUCI_PASS_HELPERS_STRINGS_H__
#define __LUCI_PASS_HELPERS_STRINGS_H__

#include "luci/Pass/QuantizationParameters.h"

#include <loco.h>

#include <vector>
#include <sstream>
#include <string>

namespace luci
{

bool in_array(const std::string &, const std::vector<std::string> &);

std::string to_string(const std::vector<std::string> &);

std::string to_lower_case(std::string);

loco::DataType str_to_dtype(const std::string &);

QuantizationGranularity str_to_granularity(const std::string &);

std::vector<std::string> split(const std::string &, const std::string &);

} // namespace luci

#endif // __LUCI_PASS_HELPERS_STRINGS_H__
