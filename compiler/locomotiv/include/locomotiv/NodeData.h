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

#ifndef _LOCOMOTIV_NODEDATA_H_
#define _LOCOMOTIV_NODEDATA_H_

#include <loco.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Shape.h>

#include <memory>

namespace locomotiv
{

/**
 * @brief Read-only no-template wrapper for 'Buffer'. Serves interface for input
 *        and output of 'Session'.
 *
 * @note Once NodeData is created, it is not modifiable.
 */
struct NodeData
{
  template <typename T> using Buffer = nncc::core::ADT::tensor::Buffer<T>;
  using Shape = nncc::core::ADT::tensor::Shape;

  virtual ~NodeData() = default;

  virtual const loco::DataType &dtype() const = 0;

  virtual const Shape *shape() const = 0;

  // TODO Support more data types
  virtual const Buffer<int32_t> *as_s32_bufptr() const = 0;
  virtual const Buffer<float> *as_f32_bufptr() const = 0;
};

/**
 * @brief Copy buffer to make NodeData
 *
 * @note NodeData is read-only. You may prepare buffer with ALL data, then call
 *       this function to make data.
 */
template <typename DT> std::unique_ptr<NodeData> make_data(const NodeData::Buffer<DT> &buffer);

} // namespace locomotiv

#endif // _LOCOMOTIV_NODEDATA_H_
