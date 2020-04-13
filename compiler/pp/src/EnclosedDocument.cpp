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

#include "pp/EnclosedDocument.h"

namespace pp
{

uint32_t EnclosedDocument::lines(void) const { return _front.lines() + _back.lines(); }

const std::string &EnclosedDocument::line(uint32_t n) const
{
  if (n < _front.lines())
  {
    return _front.line(n);
  }

  return _back.line(n - _front.lines());
}

} // namespace pp
