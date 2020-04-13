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

#ifndef __ENCO_CPP_GEN_HOST_H__
#define __ENCO_CPP_GEN_HOST_H__

#include "CppGen/MemoryContext.h"

#include <coco/IR.h>
#include <pp/MultiLineText.h>

namespace enco
{

/***
 * @brief Generate C++ code that does not depend on Anroid NN API
 */
class HostBlockCompiler
{
public:
  HostBlockCompiler(const enco::MemoryContext &mem) : _mem(mem)
  {
    // DO NOTHING
  }

public:
  std::unique_ptr<pp::MultiLineText> compile(const coco::Block *blk) const;

private:
  const enco::MemoryContext &_mem;
};

} // namespace enco

#endif // __ENCO_CPP_GEN_HOST_H__
