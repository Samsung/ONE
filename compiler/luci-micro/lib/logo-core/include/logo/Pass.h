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

#ifndef __LOGO_PASS_H__
#define __LOGO_PASS_H__

#include <loco.h>

#include <string>

namespace logo
{

class Pass
{
public:
  virtual ~Pass() = default;

public:
  virtual const char *name(void) const { return nullptr; }

public:
  /**
   * @brief  Run the pass
   *
   * @return false if there was nothing changed
   */
  virtual bool run(loco::Graph *graph) = 0;
};

std::string pass_name(const Pass *);

} // namespace logo

#endif // __LOGO_PASS_H__
