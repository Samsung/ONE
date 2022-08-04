/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONE_SERVICE_NPUD_UTIL_ENV_CONFIG_SOURCE_H__
#define __ONE_SERVICE_NPUD_UTIL_ENV_CONFIG_SOURCE_H__

#include <unordered_map>

#include "GeneralConfigSource.h"

namespace npud
{
namespace util
{

class EnvConfigSource final : public GeneralConfigSource
{
public:
  std::string get(const std::string &key) const override;

private:
  std::unordered_map<std::string, std::string> _default_attributes;
};

} // namespace util
} // namespace npud

#endif // __ONE_SERVICE_NPUD_UTIL_ENV_CONFIG_SOURCE_H__
