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

#include "NodeExecution.h"

#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <stdexcept>
#include <cassert>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

namespace
{

/**
 * @brief Get offset based on given shape and index. Assume lexical layout.
 *
 * examples)
 * For shape = {3, 4} and index = {1, 2},
 *     offset would be 6 ( = 1 * (4) + 2 )
 * For shape = {2, 3, 4} and index = {1, 0, 2},
 *     offset would be 14 ( = 1 * (3*4) + 0 *(4) + 2 )
 */
inline uint32_t offset_by_index(const Shape &shape, const Index &index)
{
  static const nncc::core::ADT::tensor::LexicalLayout l;
  return l.offset(shape, index);
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::ConstGen *constgen)
{
  uint32_t volume = 1;

  Shape shape;
  shape.resize(constgen->rank());
  for (uint32_t i = 0; i < shape.rank(); ++i)
  {
    shape.dim(i) = constgen->dim(i).value();
    volume *= shape.dim(i);
  }

  std::unique_ptr<NodeData> data = nullptr;

  switch (constgen->dtype())
  {
    case loco::DataType::S32:
    {
      assert(volume == constgen->size<loco::DataType::S32>());

      auto buf = make_buffer<int32_t, LexicalLayout>(shape);

      for (IndexEnumerator e{shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        uint32_t offset = ::offset_by_index(shape, index);
        buf.at(index) = constgen->at<loco::DataType::S32>(offset);
      }

      data = locomotiv::make_data(buf);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      assert(volume == constgen->size<loco::DataType::FLOAT32>());

      auto buf = make_buffer<float, LexicalLayout>(shape);

      for (IndexEnumerator e{shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        uint32_t offset = ::offset_by_index(shape, index);
        buf.at(index) = constgen->at<loco::DataType::FLOAT32>(offset);
      }

      data = locomotiv::make_data(buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(data != nullptr);
  annot_data(constgen, std::move(data));
  annot_domain(constgen, loco::Domain::Tensor);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::ConstGen *constgen) { execute_node(constgen); }

} // namespace locomotiv
