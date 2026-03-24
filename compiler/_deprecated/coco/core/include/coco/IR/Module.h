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

#ifndef __COCO_IR_MODULE_H__
#define __COCO_IR_MODULE_H__

#include "coco/IR/EntityManager.h"
#include "coco/IR/Block.h"
#include "coco/IR/InputList.h"
#include "coco/IR/OutputList.h"

#include <memory>

namespace coco
{

/**
 * @brief Top-level element of coco IR which represents a neural network
 */
class Module
{
public:
  Module() = default;

public:
  Module(const Module &) = delete;
  Module(Module &&) = delete;

public:
  virtual ~Module() = default;

public:
  virtual EntityManager *entity(void) = 0;
  virtual const EntityManager *entity(void) const = 0;

public:
  virtual BlockList *block(void) = 0;
  virtual const BlockList *block(void) const = 0;

public:
  virtual InputList *input(void) = 0;
  virtual const InputList *input(void) const = 0;

public:
  virtual OutputList *output(void) = 0;
  virtual const OutputList *output(void) const = 0;

public:
  static std::unique_ptr<Module> create(void);
};

} // namespace coco

#endif // __COCO_IR_MODULE_H__
