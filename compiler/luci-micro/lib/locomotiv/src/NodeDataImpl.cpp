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

#include "NodeDataImpl.h"

#include <stdex/Memory.h>

#include <cassert>

namespace
{

class NodeDataAnnotation final : public loco::NodeAnnotation
{
public:
  NodeDataAnnotation(std::unique_ptr<locomotiv::NodeData> &&data) : _data{std::move(data)}
  {
    // DO NOTHING
  }

public:
  const locomotiv::NodeData *data(void) const { return _data.get(); }

private:
  std::unique_ptr<locomotiv::NodeData> _data;
};

} // namespace

namespace locomotiv
{

template <> NodeDataImpl::NodeDataImpl(const Buffer<int32_t> &buf)
{
  _dtype = loco::DataType::S32;
  _s32.reset(new Buffer<int32_t>(buf));
  _shape = const_cast<Shape *>(&(_s32->shape()));
}

template <> NodeDataImpl::NodeDataImpl(const Buffer<float> &buf)
{
  _dtype = loco::DataType::FLOAT32;
  _f32.reset(new Buffer<float>(buf));
  _shape = const_cast<Shape *>(&(_f32->shape()));
}

void annot_data(loco::Node *node, std::unique_ptr<NodeData> &&data)
{
  node->annot(stdex::make_unique<NodeDataAnnotation>(std::move(data)));
}

const NodeData *annot_data(const loco::Node *node)
{
  if (auto annot = node->annot<NodeDataAnnotation>())
  {
    return annot->data();
  }

  return nullptr;
}

void erase_annot_data(loco::Node *node) { node->annot<NodeDataAnnotation>(nullptr); }

} // namespace locomotiv
