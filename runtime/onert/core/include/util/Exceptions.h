/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_UTIL_ONERTEXCEPTION_H__
#define __ONERT_UTIL_ONERTEXCEPTION_H__

#include <string>

namespace onert
{

class OnertException : public std::exception
{
public:
  OnertException(const std::string &msg) : _msg{msg} {}
  OnertException(const std::string &tag, const std::string &msg) : _msg{tag + " : " + msg} {}

  const char *what() const noexcept override { return _msg.c_str(); }

private:
  std::string _msg;
};

class InsufficientBufferSizeException : public OnertException
{
public:
  InsufficientBufferSizeException(const std::string &msg)
    : OnertException{"InsufficientBufferSize", msg}
  {
  }
};

} // namespace onert

#endif // __ONERT_UTIL_ONERTEXCEPTION_H__
