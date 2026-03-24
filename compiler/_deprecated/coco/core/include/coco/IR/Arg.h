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

#ifndef __COCO_IR_ARG_H__
#define __COCO_IR_ARG_H__

#include "coco/IR/Bag.h"
#include "coco/IR/ElemID.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/Layout.h>

#include <string>
#include <vector>

namespace coco
{

/**
 * @brief Base class for NN model arguments (Input/Output)
 */
class Arg
{
public:
  explicit Arg(const nncc::core::ADT::tensor::Shape &shape);

public:
  virtual ~Arg() = default;

public:
  const nncc::core::ADT::tensor::Shape &shape(void) const { return _shape; }

public:
  const std::string &name(void) const { return _name; }
  void name(const std::string &s) { _name = s; }

protected:
  virtual void onTake(Bag *) { return; }
  virtual void onRelease(Bag *) { return; }

public:
  Bag *bag(void) const { return _bag; }
  void bag(Bag *);

public:
  ElemID &at(const nncc::core::ADT::tensor::Index &);
  const ElemID &at(const nncc::core::ADT::tensor::Index &) const;

public:
  void reorder(const nncc::core::ADT::tensor::Layout &l);
  template <typename LayoutImpl> void reorder(void) { reorder(LayoutImpl{}); }

private:
  nncc::core::ADT::tensor::Shape const _shape;

private:
  std::string _name;

private:
  Bag *_bag;
  std::vector<ElemID> _map;
};

} // namespace coco

#endif // __COCO_IR_ARG_H__
