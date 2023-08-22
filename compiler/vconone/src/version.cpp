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

#include "vconone/vconone.h"

#include "version_cfg.h"

#include <sstream>

namespace vconone
{

version get_number(void)
{
  version v;
  v.v = VCONONE_VERSION;
  return v;
}

std::string get_string4(void)
{
  std::ostringstream ss;

  auto v = get_number();
  ss << unsigned(v.f.major) << "." << unsigned(v.f.minor) << "." << unsigned(v.f.patch) << "."
     << unsigned(v.f.build);

  return ss.str();
}

std::string get_string(void)
{
  std::ostringstream ss;

  auto v = get_number();
  ss << unsigned(v.f.major) << "." << unsigned(v.f.minor) << "." << unsigned(v.f.patch);

  return ss.str();
}

std::string get_copyright(void)
{
  std::string str;
  str = "Copyright (c) 2020-2023 Samsung Electronics Co., Ltd. All Rights Reserved\r\n";
  str += "Licensed under the Apache License, Version 2.0\r\n";
  str += "https://github.com/Samsung/ONE";
  return str;
}

} // namespace vconone
