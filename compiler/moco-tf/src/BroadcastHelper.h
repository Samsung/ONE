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

#ifndef __BROADCAST_HELPER_H__
#define __BROADCAST_HELPER_H__

#include <loco/IR/Node.h>
#include <loco/IR/Dimension.h>
#include <loco/IR/TensorShape.h>

#include <bino.h>
#include <fipe.h> // include "fipe.h" for clients

namespace moco
{
namespace tf
{

class BroadcastFunctor final
{
public:
  BroadcastFunctor(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::Node *build(loco::Node *in_node, const loco::TensorShape &in_shape) const;

  loco::Node *operator()(loco::Node *in_node, const loco::TensorShape &in_shape) const
  {
    return build(in_node, in_shape);
  }

  // This method assumes the followings:
  // - loco::shape_known(node) returns true, and
  // - loco::shape_get(node).domain() is loco::Domain::Tensor
  loco::Node *build(loco::Node *node) const;

  loco::Node *operator()(loco::Node *node) const { return build(node); }

private:
  loco::TensorShape _shape;
};

/**
 * @brief Create a broadcasted node
 *
 * First, append canonical.FixedReshape if rank expansion is required.
 * Then, append canonical.TensorBroadcast if dimension expansion is required
 *
 * This mimics "tf.broadcast_to" API in TensorFlow.
 */
static inline auto broadcast_to(const loco::TensorShape &shape)
  -> decltype(bino::transform_both(std::declval<BroadcastFunctor>()))
{
  return bino::transform_both(BroadcastFunctor{shape});
}

} // namespace tf
} // namespace moco

#endif // __BROADCAST_HELPER_H__
