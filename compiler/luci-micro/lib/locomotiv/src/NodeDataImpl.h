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

#ifndef _LOCOMOTIV_NODEDATAIMPL_H_
#define _LOCOMOTIV_NODEDATAIMPL_H_

#include "locomotiv/NodeData.h"

namespace locomotiv
{

/**
 * @brief An implementation of NodeData interface
 */
class NodeDataImpl final : public NodeData
{
public:
  template <typename T> using Buffer = nncc::core::ADT::tensor::Buffer<T>;
  using Shape = nncc::core::ADT::tensor::Shape;

  template <typename DT> NodeDataImpl(const Buffer<DT> &buf);

  const loco::DataType &dtype() const override { return _dtype; }

  const Shape *shape() const override { return _shape; }

  const Buffer<int32_t> *as_s32_bufptr() const override { return _s32.get(); }

  const Buffer<float> *as_f32_bufptr() const override { return _f32.get(); }

private:
  loco::DataType _dtype = loco::DataType::Unknown;
  Shape *_shape = nullptr;
  std::unique_ptr<Buffer<int32_t>> _s32 = nullptr;
  std::unique_ptr<Buffer<float>> _f32 = nullptr;
};

/// @brief Bind "NodeData" to "Node"
void annot_data(loco::Node *node, std::unique_ptr<NodeData> &&data);

/**
 * @brief Get "NodeData" for a given node
 *
 * NOTE Returns nullptr if "NodeData" is not binded yet
 */
const NodeData *annot_data(const loco::Node *node);

/// @brief Release "NodeData" bound to a given node
void erase_annot_data(loco::Node *node);

} // namespace locomotiv

#endif // _LOCOMOTIV_NODEDATAIMPL_H_
