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

#ifndef __CIRCLE_TENSOR_EXPORTER_H__
#define __CIRCLE_TENSOR_EXPORTER_H__

#include "CircleExporterUtils.h"

#include <loco/IR/Graph.h>

#include <flatbuffers/flatbuffers.h>

namespace luci
{

/**
 * @brief one time preparation for SerializedModelData
 */
void prepareModelData(flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md);

/**
 * @brief create Tensors corresponding to results of all nodes in graph
 * @param computational graph
 * @param gd information about serialized parts of model
 */
void exportOpDefinedTensors(loco::Graph *g, flatbuffers::FlatBufferBuilder &builder,
                            SerializedModelData &md, SerializedGraphData &gd);

/**
 * @brief clear temporary export information annotated to graph nodes
 */
void clearExportInfo(loco::Graph *g);

} // namespace luci

#endif // __CIRCLE_TENSOR_EXPORTER_H__
