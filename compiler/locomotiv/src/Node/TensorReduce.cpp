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
#include "Validation.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

#include <cassert>
#include <stdexcept>

namespace
{

Index reduced_index(const Index &index, const loco::TensorAxisSet &axes)
{
  Index r_index;

  r_index.resize(index.rank());
  for (uint32_t i = 0; i < index.rank(); ++i)
    r_index.at(i) = (axes.defined(i)) ? 0 : index.at(i);

  return r_index;
}

Shape reduced_shape(const Shape &shape, const loco::TensorAxisSet &axes)
{
  Shape r_shape;

  r_shape.resize(shape.rank());
  for (uint32_t i = 0; i < shape.rank(); ++i)
    r_shape.dim(i) = (axes.defined(i)) ? 1 : shape.dim(i);

  return r_shape;
}

} // namespace

namespace
{

template <typename T, loco::ReduceFunc F> struct ReduceFunction
{
  static void apply(Buffer<T> &lhs, const Buffer<T> &rhs, const loco::TensorAxisSet &axes)
  {
    throw std::runtime_error("Not supported ReduceFunc type");
  }
};

template <typename T> struct ReduceFunction<T, loco::ReduceFunc::Mean>
{
  static void apply(Buffer<T> &lhs, const Buffer<T> &rhs, const loco::TensorAxisSet &axes)
  {
    for (IndexEnumerator e{rhs.shape()}; e.valid(); e.advance())
    {
      const auto &index = e.current();
      const auto r_index = reduced_index(index, axes);

      lhs.at(r_index) += rhs.at(index);
    }

    uint32_t r_cnt = 1;
    for (uint32_t i = 0; i < rhs.shape().rank(); ++i)
      if (axes.defined(i))
        r_cnt *= rhs.shape().dim(i);

    for (IndexEnumerator e{lhs.shape()}; e.valid(); e.advance())
    {
      const auto &index = e.current();
      lhs.at(index) /= static_cast<T>(r_cnt);
    }
  }
};

template <typename T>
void apply(Buffer<T> &lhs, const Buffer<T> &rhs, const loco::TensorReduce &node)
{
  switch (node.func())
  {
    case loco::ReduceFunc::Mean:
      ReduceFunction<T, loco::ReduceFunc::Mean>::apply(lhs, rhs, *node.axes());
      break;

    // TODO Support more ReduceFunc type
    default:
      break;
  }
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TensorReduce *node)
{
  auto input_data = annot_data(node->input());
  validate(input_data, "Input not ready");
  auto input_shape = input_data->shape();
  validate(annot_domain(node->input()) == loco::Domain::Tensor,
           "Input domain of TensorReduce is not Tensor");

  std::unique_ptr<NodeData> reduce_data = nullptr;
  Shape r_shape = reduced_shape(*input_shape, *node->axes());
  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto reduce_buf = make_buffer<float, LexicalLayout>(r_shape);

      apply(reduce_buf, *input_bufptr, *node);

      reduce_data = make_data(reduce_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(reduce_data != nullptr);
  annot_data(node, std::move(reduce_data));
  annot_domain(node, annot_domain(node->input()));
}

} // namespace locomotiv
