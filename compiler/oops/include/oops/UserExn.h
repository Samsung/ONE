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

#ifndef __OOPS_USER_EXN_H__
#define __OOPS_USER_EXN_H__

#include <pepper/str.h>

#include <exception>
#include <string>
#include <map>

namespace oops
{

/**
 * @brief Exception to user
 *
 * Pass msg and one additional info, e.g.,
 *   ex) UserExn("Unsupported rank", 4);
 *   ex) UserExn("Unsupported layout", "NHWC");
 *
 * Or pass msg with attribute pairs of name & val ,
 *   ex) UserExn("Node has unsupported layout",
 *                 "Node", node->name(),
 *                 "layout", node->layout());
 */
class UserExn : public std::exception
{
public:
  UserExn() = delete;

  template <typename... Info> UserExn(const std::string &msg, Info &&...args)
  {
    std::stringstream out;

    out << "Error: " << msg + ": ";

    build_info(out, args...);

    _msg = out.str();
  }

  const char *what() const noexcept override { return _msg.c_str(); };

private:
  template <typename Attr, typename Val, typename... AttsVals>
  void build_info(std::stringstream &out, Attr &attr, Val &val, AttsVals &...args)
  {
    out << pepper::str(attr, " = ", val);
    out << ", ";

    build_info(out, args...);
  }

  template <typename Attr, typename Val>
  void build_info(std::stringstream &out, Attr &attr, Val &val)
  {
    out << pepper::str(attr, " = ", val);
  }

  void build_info(std::stringstream &)
  { /* empty */
  }

  // when only one info of string is provided
  void build_info(std::stringstream &out, const std::string &val) { out << val; }

  // when only one info of uint32_t is provided
  void build_info(std::stringstream &out, const uint32_t &val) { out << val; }

private:
  std::string _msg;
};

} // namespace oops

#endif // __OOPS_USER_EXN_H__
