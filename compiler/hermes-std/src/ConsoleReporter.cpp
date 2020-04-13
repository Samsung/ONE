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

#include "hermes/ConsoleReporter.h"

#include <iostream>

namespace hermes
{

void ConsoleReporter::notify(const hermes::Message *m)
{
  for (uint32_t n = 0; n < m->text()->lines(); ++n)
  {
    std::cout << m->text()->line(n) << std::endl;
  }
}

} // namespace hermes
