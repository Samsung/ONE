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

#ifndef __PASS_SHAPE_INFERENCE_PASS_H__
#define __PASS_SHAPE_INFERENCE_PASS_H__

#include <loco.h>
#include <logo/Pass.h>

namespace exo
{

/**
 * @brief Pass to infer shape of nodes
 */
class ShapeInferencePass : public logo::Pass
{
public:
  virtual const char *name(void) const { return "exo::ShapeInferencePass"; }

public:
  bool run(loco::Graph *graph);
};

} // namespace exo

#endif //__PASS_SHAPE_INFERENCE_PASS_H__
