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

// TODO Remove deprecated code
#if 0
#include "NodeDataImpl.h"
#include "NodeDomain.h"
#include "Validation.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

#include <cassert>
#include <stdexcept>
#endif

namespace
{

inline float relu6_ew(float val) { return val < 0.0f ? 0.0f : (val < 6.0f ? val : 6.0f); }

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::ReLU6 *relu6)
{
// TODO Remove deprecated code
#if 0
  auto input_data = annot_data(relu6->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(relu6->input()) != loco::Domain::Unknown,
           "Input domain of ReLU is Unknown");

  std::unique_ptr<NodeData> relu6_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto *shape = input_data->shape();
      auto relu6_buf = make_buffer<float, LexicalLayout>(*shape);

      for (IndexEnumerator e{*shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        relu6_buf.at(index) = relu6_ew(input_bufptr->at(index));
      }

      relu6_data = make_data(relu6_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(relu6_data != nullptr);
  annot_data(relu6, std::move(relu6_data));
  annot_domain(relu6, annot_domain(relu6->input()));
#endif

  struct Func final : public UnaryFunc
  {
    float apply(float v) const final { return relu6_ew(v); }
  };

  Func f;

  eltwise_unary(relu6, f);
}

} // namespace locomotiv
