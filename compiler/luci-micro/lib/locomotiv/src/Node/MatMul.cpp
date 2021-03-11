/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include <cassert>
#include <stdexcept>

namespace
{
using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

/**
 * @brief Calculate Matrix Multiplication
 */
template <typename T> Buffer<T> calc_mat_mul(const Buffer<T> *lhs_buf, const Buffer<T> *rhs_buf)
{
  const auto lhs_shape = lhs_buf->shape();
  const auto rhs_shape = rhs_buf->shape();

  assert(lhs_shape.rank() == 2 && "lhs rank must be 2");
  assert(rhs_shape.rank() == 2 && "rhs rank must be 2");
  // lhs width should be the same as rhs height
  assert(lhs_shape.dim(1) == rhs_shape.dim(0) && "height/width mismatch");

  const uint32_t lhs_height = lhs_shape.dim(0);
  const uint32_t lhs_width = lhs_shape.dim(1);

  const uint32_t rhs_width = rhs_shape.dim(1);

  const uint32_t output_height = lhs_height;
  const uint32_t output_width = rhs_width;

  Shape output_shape{output_height, output_width};
  auto output_buf = make_buffer<T, LexicalLayout>(output_shape);

  for (uint32_t out_y = 0; out_y < output_height; ++out_y)
  {
    for (uint32_t out_x = 0; out_x < output_width; ++out_x)
    {
      T total = static_cast<T>(0); // accumulator
      // Accumulate through axis
      for (uint32_t axis = 0; axis < lhs_width; ++axis)
      {
        total += lhs_buf->at(Index({out_y, axis})) * rhs_buf->at(Index({axis, out_x}));
      }
      // Set output value
      output_buf.at(Index({out_y, out_x})) = total;
    }
  }

  return output_buf;
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::MatMul *mat_mul)
{
  auto lhs_data = annot_data(mat_mul->lhs());
  auto rhs_data = annot_data(mat_mul->rhs());

  validate(lhs_data, "Can't find left matrix data of MatMul");
  validate(lhs_data->shape()->rank() == 2, "lhs rank must be 2");

  validate(rhs_data, "Can't find right matrix data of MatMul");
  validate(rhs_data->shape()->rank() == 2, "rhs rank must be 2");

  validate(annot_domain(mat_mul->lhs()) == loco::Domain::Matrix,
           "Left matrix of MatMul is not a Matrix");
  validate(annot_domain(mat_mul->rhs()) == loco::Domain::Matrix,
           "Right matrix of MatMul is not a Matrix");

  std::unique_ptr<NodeData> mat_mul_result = nullptr;

  if (lhs_data->dtype() == loco::DataType::FLOAT32 && rhs_data->dtype() == loco::DataType::FLOAT32)
  {
    const auto lhs_buf = lhs_data->as_f32_bufptr();
    const auto rhs_buf = rhs_data->as_f32_bufptr();

    auto mat_mul_buf = calc_mat_mul<float>(lhs_buf, rhs_buf);

    mat_mul_result = make_data(mat_mul_buf);
  }
  else if (lhs_data->dtype() == loco::DataType::S32 && rhs_data->dtype() == loco::DataType::S32)
  {
    const auto lhs_buf = lhs_data->as_s32_bufptr();
    const auto rhs_buf = rhs_data->as_s32_bufptr();

    auto mat_mul_buf = calc_mat_mul<int32_t>(lhs_buf, rhs_buf);

    mat_mul_result = make_data(mat_mul_buf);
  }
  else
    throw std::runtime_error("NYI for these DataTypes");

  assert(mat_mul_result != nullptr);

  annot_data(mat_mul, std::move(mat_mul_result));
  annot_domain(mat_mul, loco::Domain::Matrix);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::MatMul *mat_mul) { execute_node(mat_mul); }

} // namespace locomotiv
