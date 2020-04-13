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

#include "env.h"

#include "misc/environment.h"

//
// Integer variable
//
IntVar::IntVar(const std::string &name, int32_t value) : _value{value}
{
  nnfw::misc::env::IntAccessor{name}.access(_value);
}

//
// Float variable
//
FloatVar::FloatVar(const std::string &name, float value) : _value{value}
{
  nnfw::misc::env::FloatAccessor{name}.access(_value);
}

//
// String variable
//
#include <cstdlib>

StrVar::StrVar(const std::string &name, const std::string &value) : _value{value}
{
  auto env = std::getenv(name.c_str());

  if (env)
  {
    _value = std::string{env};
  }
}
