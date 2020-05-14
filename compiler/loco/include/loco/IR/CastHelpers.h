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

#ifndef __LOCO_IR_CAST_HELPERS_H__
#define __LOCO_IR_CAST_HELPERS_H__

#include <string>
#include <stdexcept>
#include <typeinfo>

namespace loco
{

// TODO move to somewhere appropriate
template <typename T, typename ARG> T _must_cast(ARG arg)
{
  auto cast_arg = dynamic_cast<T>(arg);
  if (cast_arg == nullptr)
  {
    std::string msg = "loco::must_cast() failed to cast: ";
    msg += typeid(T).name();
    throw std::invalid_argument(msg.c_str());
  }
  return cast_arg;
}

} // namespace loco

#endif // __LOCO_IR_CAST_HELPERS_H__
