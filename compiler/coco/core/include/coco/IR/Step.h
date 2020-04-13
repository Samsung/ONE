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

#ifndef __COCO_IR_STEP_H__
#define __COCO_IR_STEP_H__

#include "coco/IR/Op.forward.h"
#include "coco/IR/Instr.forward.h"

namespace coco
{

/**
 * @brief A Step denotes the edge between Op and Instr
 */
class Step final
{
public:
  explicit Step(Instr *instr) : _instr{instr}
  {
    // DO NOTHING
  }

public:
  ~Step() { op(nullptr); }

public:
  Op *op(void) const { return _op; }
  void op(Op *o);

public:
  Instr *instr(void) const { return _instr; }

private:
  Op *_op = nullptr;
  Instr *_instr = nullptr;
};

} // namespace coco

#endif // __COCO_IR_STEP_H__
