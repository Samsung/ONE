/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __OP_PADDING_H__
#define __OP_PADDING_H__

#include <coco/IR/Padding2D.h>
#include <nncc/core/ADT/tensor/Shape.h>

#include <schema_generated.h>

using namespace nncc::core::ADT;

namespace tflimport
{

coco::Padding2D pool2D_padding(const tflite::Pool2DOptions *options, const tensor::Shape &ifm_shape,
                               const int filter_w, const int filter_h);

coco::Padding2D conv2D_padding(const tflite::Conv2DOptions *options, const tensor::Shape &ifm_shape,
                               const tensor::Shape &kernel_shape);

coco::Padding2D depthwiseConv2D_padding(const tflite::DepthwiseConv2DOptions *options,
                                        const tensor::Shape &ifm_shape,
                                        const tensor::Shape &kernel_shape);

} // namespace tflimport

#endif // __OP_PADDING_H__
