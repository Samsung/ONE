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

#ifndef __SOUSCHEF_RANGED_ARGUMENTS_H__
#define __SOUSCHEF_RANGED_ARGUMENTS_H__

#include "Arguments.h"

#include <string>

namespace souschef
{

template <typename InputIt> class RangedArguments : public Arguments
{
public:
  RangedArguments(InputIt beg, InputIt end) : _beg{beg}, _end{end}
  {
    // DO NOTHING
  }

public:
  uint32_t count(void) const override { return _end - _beg; }

public:
  const std::string &value(uint32_t n) const override { return *(_beg + n); }

private:
  InputIt _beg;
  InputIt _end;
};

template <typename InputIt> RangedArguments<InputIt> ranged_arguments(InputIt beg, InputIt end)
{
  return RangedArguments<InputIt>{beg, end};
}

} // namespace souschef

#endif // __SOUSCHEF_RANGED_ARGUMENTS_H__
