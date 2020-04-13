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

#ifndef __PP_MULTI_LINE_TEXT_H__
#define __PP_MULTI_LINE_TEXT_H__

#include <string>

#include <cstdint>

namespace pp
{

struct MultiLineText
{
  virtual ~MultiLineText() = default;

  virtual uint32_t lines(void) const = 0;
  virtual const std::string &line(uint32_t n) const = 0;
};

} // namespace pp

#endif // __PP_MULTI_LINE_TEXT_H__
