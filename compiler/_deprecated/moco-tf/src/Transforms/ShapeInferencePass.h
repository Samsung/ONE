/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_TF_SHAPE_INFERENCE_PASS_H__
#define __MOCO_TF_SHAPE_INFERENCE_PASS_H__

#include "Transform.h"

#include <loco.h>

namespace moco
{
namespace tf
{

/**
 * @brief  Run shape inference to the graph
 */
class ShapeInferencePass : public Transform
{
public:
  const char *name(void) const final { return "ShapeInferencePass"; }

public:
  bool run(loco::Graph *graph) override;
};

} // namespace tf
} // namespace moco

#endif // __MOCO_TF_SHAPE_INFERENCE_PASS_H__
