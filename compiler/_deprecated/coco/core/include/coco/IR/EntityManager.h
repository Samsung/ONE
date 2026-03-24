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

#ifndef __COCO_IR_ENTITY_MANAGER_H__
#define __COCO_IR_ENTITY_MANAGER_H__

#include "coco/IR/BagManager.h"
#include "coco/IR/ObjectManager.h"

#include "coco/IR/OpManager.h"
#include "coco/IR/InstrManager.h"
#include "coco/IR/BlockManager.h"

#include "coco/IR/InputManager.h"
#include "coco/IR/OutputManager.h"

namespace coco
{

/**
 * @brief Meta (lifetime) manager interface
 *
 * EntityManager is referred as meta manager as it is a gateway to other
 * managers.
 */
struct EntityManager
{
  virtual ~EntityManager() = default;

  virtual BagManager *bag(void) = 0;
  virtual const BagManager *bag(void) const = 0;

  virtual ObjectManager *object(void) = 0;
  virtual const ObjectManager *object(void) const = 0;

  virtual OpManager *op(void) = 0;
  virtual const OpManager *op(void) const = 0;

  virtual InstrManager *instr(void) = 0;
  virtual const InstrManager *instr(void) const = 0;

  virtual BlockManager *block(void) = 0;
  virtual const BlockManager *block(void) const = 0;

  virtual InputManager *input(void) = 0;
  virtual const InputManager *input(void) const = 0;

  virtual OutputManager *output(void) = 0;
  virtual const OutputManager *output(void) const = 0;
};

} // namespace coco

#endif // __COCO_IR_ENTITY_MANAGER_H__
