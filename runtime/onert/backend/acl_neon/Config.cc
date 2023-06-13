/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Config.h"

#include <util/ConfigSource.h>

namespace onert
{
namespace backend
{
namespace acl_neon
{

bool Config::initialize() { return true; }

ir::Layout Config::supportLayout(const ir::IOperation &, ir::Layout frontend_layout)
{
  const std::string acl_layout_str = util::getConfigString(util::config::ACL_LAYOUT);
  if (acl_layout_str == "NHWC")
  {
    return ir::Layout::NHWC;
  }
  else if (acl_layout_str == "NCHW")
  {
    return ir::Layout::NCHW;
  }

  return frontend_layout;
}

} // namespace acl_neon
} // namespace backend
} // namespace onert
