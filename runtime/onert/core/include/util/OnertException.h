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

#include <iostream>

namespace onert
{
namespace util
{

class OnertException : public std::exception
{
  std::string message;

public:
  OnertException(std::string _m) : message(_m) {}

  virtual const char *what() const throw() { return message.c_str(); }
};

class ExecutionFailedException : public OnertException
{
public:
  ExecutionFailedException(std::string _m) : OnertException("ExecutionFailedException : " + _m) {}
};

class FileNotFoundException : public OnertException
{
public:
  FileNotFoundException(std::string _m) : OnertException("FileNotFoundException : " + _m) {}
};

class NotSupportedTypeException : public OnertException
{
public:
  NotSupportedTypeException(std::string _m) : OnertException("NotSupportedTypeException : " + _m) {}
};

class InvalidValueException : public OnertException
{
public:
  InvalidValueException(std::string _m) : OnertException("InvalidValueException : " + _m) {}
};

class NotImplementedException : public OnertException
{
public:
  NotImplementedException(std::string _m) : OnertException("NotImplementedException : " + _m) {}
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_ONERTEXCEPTION_H__
