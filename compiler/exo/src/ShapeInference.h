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

#ifndef __SHAPE_INFERENCE_H__
#define __SHAPE_INFERENCE_H__

#include "ExporterUtils.h"

#include <loco/IR/Nodes.h>

namespace exo
{

/**
 * @brief Get the shape of each node as a node annotation
 *
 * HOW TO USE
 *
 *   ShapeInference::get(g->nodes()->at(..));
 */
struct ShapeInference
{
  static ShapeDescription get(loco::Node *node);
};

} // namespace exo

#endif // __SHAPE_INFERENCE_H__
