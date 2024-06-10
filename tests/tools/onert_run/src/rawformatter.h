/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_RUN_RAWFORMATTER_H__
#define __ONERT_RUN_RAWFORMATTER_H__

#include "allocation.h"
#include "types.h"

#include <string>
#include <vector>

struct nnfw_session;

namespace onert_run
{
class RawFormatter
{
public:
  RawFormatter() = default;
  void loadInputs(const std::string &filename, std::vector<Allocation> &inputs);
  void dumpOutputs(const std::string &filename, const std::vector<Allocation> &outputs);
  void dumpInputs(const std::string &filename, const std::vector<Allocation> &inputs);
};
} // namespace onert_run

#endif // __ONERT_RUN_RAWFORMATTER_H__
