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

#ifndef __LUCI_CIRCLE_TYPE_INFERENCE_H__
#define __LUCI_CIRCLE_TYPE_INFERENCE_H__

#include <loco/IR/Nodes.h>

#include <mio/circle/schema_generated.h>

namespace luci
{

/**
 * @brief Get the type of each node as NodeAnnotation
 *
 * HOW TO USE
 *
 *   TypeInference::get(g->nodes()->at(0));
 *   TypeInference::get(g->nodes()->at(...));
 */
struct TypeInference
{
  static circle::TensorType get(loco::Node *node);
};

} // namespace luci

#endif // __LUCI_CIRCLE_TYPE_INFERENCE_H__
