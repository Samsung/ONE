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

#include "hermes/EnvConfig.h"

#include <pepper/strcast.h>

namespace hermes
{

EnvConfig<EnvFormat::BooleanNumber>::EnvConfig(const EnvName &name)
{
  auto s = std::getenv(name.c_str());
  _enabled = (pepper::safe_strcast<int>(s, 0 /* DISABLE BY DEFAULT */) != 0);
}

void EnvConfig<EnvFormat::BooleanNumber>::configure(const Source *, SourceSetting &setting) const
{
  if (_enabled)
  {
    // Enable all the sources
    setting.accept_all();
  }
  else
  {
    // Disable all the sources
    setting.reject_all();
  }
}

} // namespace hermes
