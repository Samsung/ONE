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

#ifndef __PEPPER_STR_H__
#define __PEPPER_STR_H__

#include <ostream>
#include <sstream>

#include <string>

namespace pepper
{
namespace details
{

template <typename... Arg> void str_impl(std::ostream &os, Arg &&...args);

template <> inline void str_impl(std::ostream &)
{
  // DO NOTHING
  return;
}

template <typename Arg> inline void str_impl(std::ostream &os, Arg &&arg)
{
  os << std::forward<Arg>(arg);
}

template <typename Arg, typename... Args>
inline void str_impl(std::ostream &os, Arg &&arg, Args &&...args)
{
  str_impl(os, std::forward<Arg>(arg));
  str_impl(os, std::forward<Args>(args)...);
}

} // namespace details
} // namespace pepper

namespace pepper
{

template <typename... Args> static inline std::string str(Args &&...args)
{
  std::stringstream ss;
  details::str_impl(ss, std::forward<Args>(args)...);
  return ss.str();
}

} // namespace pepper

#endif // __PEPPER_STR_H__
