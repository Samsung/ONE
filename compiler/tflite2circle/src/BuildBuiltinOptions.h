/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __BUILD_BUITIN_OPTIONS_H__
#define __BUILD_BUITIN_OPTIONS_H__

#include "DataLookup.h"

namespace tflite2circle
{

flatbuffers::Offset<circle::Conv2DOptions>
build_circle_Conv2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::DepthwiseConv2DOptions>
build_circle_DepthwiseConv2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::Pool2DOptions>
build_circle_Pool2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::ConcatenationOptions>
build_circle_ConcatenationOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::AddOptions> build_circle_AddOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op);

flatbuffers::Offset<circle::ReshapeOptions>
build_circle_ReshapeOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::PadOptions> build_circle_PadOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op);

flatbuffers::Offset<circle::SubOptions> build_circle_SubOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op);

flatbuffers::Offset<circle::DivOptions> build_circle_DivOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op);

flatbuffers::Offset<circle::SoftmaxOptions>
build_circle_SoftmaxOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::FullyConnectedOptions>
build_circle_FullyConnectedOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

flatbuffers::Offset<circle::ArgMaxOptions>
build_circle_ArgMaxOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op);

} // namespace tflite2circle

#endif // __BUILD_BUITIN_OPTIONS_H__
