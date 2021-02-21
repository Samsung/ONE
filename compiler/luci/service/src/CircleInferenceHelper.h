/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_INFERENCE_HELPER_H__
#define __CIRCLE_INFERENCE_HELPER_H__

#include <luci/IR/CircleNodes.h>

namespace luci
{

struct CircleIfOutGraphs
{
  loco::GraphOutput *then_graph_output;
  loco::GraphOutput *else_graph_output;
};

/**
 * @brief Return 'THEN' and 'ELSE' output graphs for input node
 */
CircleIfOutGraphs get_out_graphs(const luci::CircleIfOut *node);

} // namespace luci

#endif // __CIRCLE_INFERENCE_HELPER_H__
