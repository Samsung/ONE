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

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::num_elements;

#include <cassert>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace
{

using namespace locomotiv;

void execute_node(loco::Reshape<loco::ReshapeType::Fixed> *reshape)
{
  auto input_data = annot_data(reshape->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(reshape->input()) == loco::Domain::Tensor,
           "Input domain of Reshape is not Tensor");

  std::unique_ptr<NodeData> reshape_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto *input_shape = input_data->shape();

      using Shape = nncc::core::ADT::tensor::Shape;
      std::unique_ptr<Shape> output_shape(new Shape());

      output_shape->resize(reshape->rank());
      for (uint32_t axis = 0; axis < output_shape->rank(); ++axis)
      {
        output_shape->dim(axis) = reshape->dim(axis).value();
      }

      auto reshape_bufptr = make_buffer<float, LexicalLayout>(*output_shape);

      float *input_ptr = const_cast<float *>(input_bufptr->base());
      uint64_t input_len = num_elements(*input_shape) * sizeof(float);

      float *output_ptr = reshape_bufptr.base();

      assert(input_len == num_elements(*output_shape) * sizeof(float));
      memcpy(output_ptr, input_ptr, input_len);

      reshape_data = make_data(reshape_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(reshape_data != nullptr);
  annot_data(reshape, std::move(reshape_data));
  annot_domain(reshape, annot_domain(reshape->input()));
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Reshape<loco::ReshapeType::Fixed> *reshape)
{
  execute_node(reshape);
}

} // namespace locomotiv
