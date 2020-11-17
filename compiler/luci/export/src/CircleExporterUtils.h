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

#ifndef __CIRCLE_EXPORTER_UTILS_H__
#define __CIRCLE_EXPORTER_UTILS_H__

#include "SerializedData.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/ShapeDescription.h>

#include <loco.h>

#include <mio/circle/schema_generated.h>

namespace luci
{

circle::ActivationFunctionType to_circle_actfunc(luci::FusedActFunc func);
circle::TensorType to_circle_tensortype(loco::DataType type);
circle::MirrorPadMode to_circle_mirrorpadmode(luci::MirrorPadMode mode);
circle::FullyConnectedOptionsWeightsFormat
to_circle_weightsformat(luci::CircleFullyConnected::WeightsFormat format);
circle::DimensionType to_circle_dimensiontype(luci::DimensionType type);
flatbuffers::Offset<void> to_circle_sparse_index_vector(flatbuffers::FlatBufferBuilder &fb,
                                                        const SparseIndexVector &sparse_idx_vec);
circle::SparseIndexVector to_circle_sparse_index_vector_type(luci::SparseIndexVectorType type);

} // namespace luci

namespace luci
{

circle::Padding getOpPadding(const loco::Padding2D *pad, const loco::Stride<2> *stride,
                             const ShapeDescription &ifm, const ShapeDescription &ofm);
circle::Padding getOpPadding(const luci::Padding pad);

using CircleTensorIndex = int32_t;

void set_tensor_index(loco::Node *node, const CircleTensorIndex &tensor_id);
CircleTensorIndex get_tensor_index(loco::Node *node);

} // namespace luci

#endif // __CIRCLE_EXPORTER_UTILS_H__
