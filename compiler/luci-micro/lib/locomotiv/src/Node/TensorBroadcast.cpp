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
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

#include <cassert>
#include <stdexcept>

namespace
{

using namespace locomotiv;

void execute_node(loco::TensorBroadcast *tensor_broadcast)
{
  auto input_data = annot_data(tensor_broadcast->input());

  // Calculate output shape
  Shape input_shape = *(input_data->shape());

  // TODO Reuse "ShapeInferenceService"
  Shape output_shape;

  output_shape.resize(input_shape.rank());
  for (uint32_t axis = 0; axis < input_shape.rank(); ++axis)
  {
    if (tensor_broadcast->mapping()->defined(axis))
    {
      assert(input_shape.dim(axis) == 1); // Required by TensorBroadcast definition
      output_shape.dim(axis) = tensor_broadcast->mapping()->dim(axis).value();
    }
    else
    {
      output_shape.dim(axis) = input_shape.dim(axis);
    }
  }

  assert(input_shape.rank() == output_shape.rank());

  uint32_t const rank = input_shape.rank();

  std::unique_ptr<NodeData> output_data = nullptr;

  switch (input_data->dtype())
  {
    // TODO Use type-generic implementation!
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto output_buf = make_buffer<float, LexicalLayout>(output_shape);

      for (IndexEnumerator e{output_shape}; e.valid(); e.advance())
      {
        auto input_index = e.current();
        const auto &output_index = e.current();

        for (uint32_t axis = 0; axis < rank; ++axis)
        {
          if (tensor_broadcast->mapping()->defined(axis))
          {
            input_index.at(axis) = 0;
          }
        }

        output_buf.at(output_index) = input_bufptr->at(input_index);
      }

      output_data = make_data(output_buf);
      break;
    }
    default:
      throw std::runtime_error("Not yet supported");
  }

  assert(output_data != nullptr);
  annot_data(tensor_broadcast, std::move(output_data));
  annot_domain(tensor_broadcast, loco::Domain::Tensor);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TensorBroadcast *tensor_broadcast)
{
  execute_node(tensor_broadcast);
}

} // namespace locomotiv
