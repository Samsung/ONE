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

#ifndef __COCO_IR_OP_MANAGER_H__
#define __COCO_IR_OP_MANAGER_H__

#include "coco/IR/Op.h"
#include "coco/IR/Ops.h"

#include "coco/IR/Instr.forward.h"

#include "coco/IR/Object.forward.h"

#include "coco/IR/EntityBuilder.h"

#include "coco/ADT/PtrManager.h"

namespace coco
{

class OpManager final : public PtrManager<Op>, public EntityBuilder
{
public:
  OpManager(Module *m = nullptr) { module(m); }

public:
  ~OpManager();

public:
  template <typename T> T *create(void);

public:
  /**
   * @brief Destroy (= deallocate) a Op instance
   *
   * NOTE destroy(op) WILL NOT update op->parent(). Client SHOULD detach op before destroy(op) call
   */
  void destroy(Op *);

  /**
   * @brief Destroy a Op tree
   *
   * @require op->parent() == nullptr && op->up() == nullptr
   */
  void destroy_all(Op *);
};

} // namespace coco

#endif // __COCO_IR_OP_MANAGER_H__
