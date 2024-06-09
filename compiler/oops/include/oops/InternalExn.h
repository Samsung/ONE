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

#ifndef __OOPS_INTERNAL_EXN_H__
#define __OOPS_INTERNAL_EXN_H__

#include <exception>
#include <string>
#include <cstdint>

/// @ brief throw internal exception with message
#define INTERNAL_EXN(msg) throw oops::InternalExn(__FILE__, __LINE__, msg)

/// @ brief throw internal exception with message and value
#define INTERNAL_EXN_V(msg, val) throw oops::InternalExn(__FILE__, __LINE__, msg, val)

namespace oops
{

template <typename T> uint32_t to_uint32(T a) { return static_cast<uint32_t>(a); }

/**
 * @brief Exception caused by internal error
 *
 * Note: Please use the above MACROs
 */
class InternalExn : public std::exception
{
public:
  InternalExn(const char *filename, const int line, const std::string &msg)
    : _filename(filename), _line(to_uint32(line)), _msg(msg)
  {
    construct_full_msg();
  }

  explicit InternalExn(const char *filename, const int line, const std::string &msg, uint32_t val)
    : _filename(filename), _line(to_uint32(line)), _msg(msg + ": " + std::to_string(val))
  {
    construct_full_msg();
  }

  explicit InternalExn(const char *filename, const int line, const std::string &msg,
                       const std::string &val)
    : _filename(filename), _line(to_uint32(line)), _msg(msg + ": " + val)
  {
    construct_full_msg();
  }

  const char *what() const noexcept override { return _full_msg.c_str(); }

private:
  const std::string _filename;
  const uint32_t _line;
  const std::string _msg;

private:
  void construct_full_msg()
  {
    _full_msg =
      "Internal Exception. " + _msg + " [" + _filename + ":" + std::to_string(_line) + "]";
  }

  std::string _full_msg;
};

} // namespace oops

#endif // __OOPS_INTERNAL_EXN_H__
