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

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <iostream>

class VLogging
{
public:
  static VLogging &access(void);
  bool enabled() const { return _enabled; }
  std::ostream &stream(void);

private:
  VLogging();

private:
  bool _enabled;
};

#define LOG(...) std::cout << std::endl
#define VLOG(...)                   \
  if (VLogging::access().enabled()) \
    (VLogging::access().stream() << std::endl)
#define NYI(module) std::cout << "NYI : '" << module << "' is not supported now." << std::endl;

#endif // __LOGGING_H__
