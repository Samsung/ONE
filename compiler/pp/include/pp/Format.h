/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __PP_FORMAT_H__
#define __PP_FORMAT_H__

#include <ostream>
#include <sstream>

namespace pp
{

template <typename Arg> static inline void _fmt(std::ostream &os, const Arg &arg) { os << arg; }
template <typename Arg, typename... Args>
static inline void _fmt(std::ostream &os, const Arg &arg, const Args &...args)
{
  _fmt(os, arg);
  _fmt(os, args...);
}

template <typename... Args> static inline std::string fmt(const Args &...args)
{
  std::stringstream ss;
  _fmt(ss, args...);
  return ss.str();
}

} // namespace pp

#endif // __PP_FORMAT_H__
