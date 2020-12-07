/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_SHAPE_DESCRIPTION_H__
#define __LUCI_SHAPE_DESCRIPTION_H__

#include <loco/IR/PermutingCodec.h>
#include <loco/IR/NodeShape.h>

#include <luci/IR/CircleNodes.h>

#include <cstdint>
#include <vector>

namespace luci
{

struct ShapeDescription
{
  std::vector<int32_t> _dims;
  bool _rank_known;
};

// TODO remove these when CircleDialect is fully functioal
ShapeDescription to_shape_description(const luci::CircleNode *node);
ShapeDescription to_shape_description(const loco::TensorShape &shape);
ShapeDescription to_shape_description(const loco::FeatureShape &shape);
ShapeDescription to_shape_description(const loco::FilterShape &shape);
ShapeDescription to_shape_description(const loco::BiasShape &shape);
ShapeDescription to_shape_description(const loco::MatrixShape &shape);
ShapeDescription to_shape_description(const loco::NodeShape &shape);

template <typename Permutation> inline bool isNHWC(Permutation *perm);

template <> inline bool isNHWC(loco::Permutation<loco::Domain::Feature> *perm)
{
  return perm->axis(loco::FeatureAxis::Count) == 0 && perm->axis(loco::FeatureAxis::Height) == 1 &&
         perm->axis(loco::FeatureAxis::Width) == 2 && perm->axis(loco::FeatureAxis::Depth) == 3;
}

template <> inline bool isNHWC(loco::Permutation<loco::Domain::Filter> *perm)
{
  return perm->axis(loco::FilterAxis::Count) == 0 && perm->axis(loco::FilterAxis::Height) == 1 &&
         perm->axis(loco::FilterAxis::Width) == 2 && perm->axis(loco::FilterAxis::Depth) == 3;
}

} // namespace luci

#endif // __LUCI_SHAPE_DESCRIPTION_H__
