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

#ifndef __ENCO_PIPELINE_H__
#define __ENCO_PIPELINE_H__

#include "Pass.h"

#include <memory>
#include <vector>
#include <cstdint>

namespace enco
{

class Pipeline
{
public:
  uint32_t size(void) const { return _passes.size(); }

public:
  const Pass &at(uint32_t n) const { return *(_passes.at(n)); }

public:
  void append(std::unique_ptr<Pass> &&pass) { _passes.emplace_back(std::move(pass)); }

private:
  std::vector<std::unique_ptr<Pass>> _passes;
};

} // namespace enco

#endif // __ENCO_PIPELINE_H__
