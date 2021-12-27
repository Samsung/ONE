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

#ifndef __CIRCLE_OPERATION_EXPORTER_H__
#define __CIRCLE_OPERATION_EXPORTER_H__

#include "SerializedData.h"

#include <loco/IR/Graph.h>

namespace luci
{

/**
 * @brief create Operators corresponding to model nodes
 * @param nodes container with nodes
 * @param gd information about serializer parts of model
 */
void exportNodes(loco::Graph *g, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md,
                 SerializedGraphData &gd);

} // namespace luci

#endif // __CIRCLE_OPERATION_EXPORTER_H__
