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

#ifndef __ENCO_CODE_H__
#define __ENCO_CODE_H__

#include "ANN/Context.h"

#include <coco/IR/Module.h>
#include <coco/IR/Data.h>

namespace enco
{

struct Code
{
public:
  Code(coco::Module *module, coco::Data *data) : _module{module}, _data{data}
  {
    // DO NOTHING
  }

public:
  coco::Module *module(void) const { return _module; }
  coco::Data *data(void) const { return _data; }

private:
  coco::Module *const _module;
  coco::Data *const _data;
};

} // namespace enco

#endif // __ENCO_CODE_H__
