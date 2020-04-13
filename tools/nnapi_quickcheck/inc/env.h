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

#ifndef __ENV_UTILS_H__
#define __ENV_UTILS_H__

#include <string>

#include <cstdint>

class IntVar
{
public:
  IntVar(const std::string &name, int32_t value);

public:
  int32_t operator()(void) const { return _value; }

private:
  int32_t _value;
};

class FloatVar
{
public:
  FloatVar(const std::string &name, float value);

public:
  float operator()(void) const { return _value; }

private:
  float _value;
};

class StrVar
{
public:
  StrVar(const std::string &name, const std::string &value);

public:
  const std::string &operator()(void) const { return _value; }

private:
  std::string _value;
};

#endif // __ENV_UTILS_H__
