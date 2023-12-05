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

#include "ShapeConverter.h"

namespace onert
{
namespace exec
{

ir::Shape convertShape(const ir::Shape &shape, ir::Layout src_layout, ir::Layout dst_layout)
{
  if (shape.rank() != 4)
    return shape;

  if (src_layout == dst_layout || src_layout == ir::Layout::UNKNOWN ||
      dst_layout == ir::Layout::UNKNOWN)
    return shape;

  if ((src_layout == ir::Layout::NCHW) && (dst_layout == ir::Layout::NHWC))
  {
    const ir::Shape &src_NCHW = shape;
    ir::Shape dst_NHWC(4);
    dst_NHWC.dim(0) = src_NCHW.dim(0); // N
    dst_NHWC.dim(1) = src_NCHW.dim(2); // H
    dst_NHWC.dim(2) = src_NCHW.dim(3); // W
    dst_NHWC.dim(3) = src_NCHW.dim(1); // C

    return dst_NHWC;
  }

  if ((src_layout == ir::Layout::NHWC) && (dst_layout == ir::Layout::NCHW))
  {
    const ir::Shape &src_NHWC = shape;
    ir::Shape dst_NCHW(4);
    dst_NCHW.dim(0) = src_NHWC.dim(0); // N
    dst_NCHW.dim(1) = src_NHWC.dim(3); // C
    dst_NCHW.dim(2) = src_NHWC.dim(1); // H
    dst_NCHW.dim(3) = src_NHWC.dim(2); // W

    return dst_NCHW;
  }

  throw std::runtime_error("Should not reach here");
}

} // namespace exec
} // namespace onert
