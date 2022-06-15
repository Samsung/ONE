/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONE_SERVICE_NPUD_CORE_CORE_H__
#define __ONE_SERVICE_NPUD_CORE_CORE_H__

#include "EventLoop.h"

class Core
{
public:
  static void run(void);
  static void stop(void);

private:
  Core();

  static Core &instance(void)
  {
    static Core core;
    return core;
  }

  static EventLoop _loop;
};

#endif // __ONE_SERVICE_NPUD_CORE_CORE_H__
