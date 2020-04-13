/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NEURUN_GRAPH_SHAPE_INFERENCE_H__
#define __NEURUN_GRAPH_SHAPE_INFERENCE_H__

#include "ir/operation/AvgPool2D.h"
#include "ir/operation/Concat.h"
#include "ir/operation/MaxPool2D.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/Operands.h"
#include "ir/Index.h"
#include "ir/Layout.h"

namespace neurun
{
namespace shape_inference
{

using Shapes = std::vector<ir::Shape>;

Shapes inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape);

Shapes inferAvgPoolShape(const ir::Shape &in_shape, const ir::operation::AvgPool2D::Param &param,
                         ir::Layout layout = ir::Layout::NHWC);

Shapes inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param);

Shapes inferMaxPoolShape(const ir::Shape &in_shape, const ir::operation::MaxPool2D::Param &param,
                         ir::Layout layout = ir::Layout::NHWC);

Shapes inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                        const ir::operation::Conv2D::Param &param,
                        ir::Layout layout = ir::Layout::NHWC);

Shapes inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                                 const ir::operation::DepthwiseConv2D::Param &param,
                                 ir::Layout layout = ir::Layout::NHWC);

Shapes inferFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &ker_shape);

} // namespace shape_inference
} // namespace neurun

#endif // __NEURUN_GRAPH_SHAPE_INFERENCE_H__
