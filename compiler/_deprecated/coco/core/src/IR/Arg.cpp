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

#include "coco/IR/Arg.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <cassert>

namespace
{

const nncc::core::ADT::tensor::LexicalLayout l;

} // namespace

namespace coco
{

Arg::Arg(const nncc::core::ADT::tensor::Shape &shape) : _shape{shape}, _bag{nullptr}
{
  _map.resize(nncc::core::ADT::tensor::num_elements(shape));
}

void Arg::bag(Bag *bag)
{
  if (_bag != nullptr)
  {
    onRelease(_bag);
    _bag = nullptr;
  }

  assert(_bag == nullptr);

  if (bag != nullptr)
  {
    _bag = bag;
    onTake(_bag);
  }
}

ElemID &Arg::at(const nncc::core::ADT::tensor::Index &index)
{
  return _map.at(l.offset(_shape, index));
}

const ElemID &Arg::at(const nncc::core::ADT::tensor::Index &index) const
{
  return _map.at(l.offset(_shape, index));
}

void Arg::reorder(const nncc::core::ADT::tensor::Layout &l)
{
  using nncc::core::ADT::tensor::IndexEnumerator;

  for (IndexEnumerator e{shape()}; e.valid(); e.advance())
  {
    const auto offset = static_cast<uint32_t>(l.offset(shape(), e.current()));

    at(e.current()) = coco::ElemID{offset};
  }
}

} // namespace coco
