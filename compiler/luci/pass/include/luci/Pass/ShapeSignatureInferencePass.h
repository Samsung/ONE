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

#ifndef __LUCI_SHAPE_SIGNATURE_INFERENCE_PASS_H__
#define __LUCI_SHAPE_SIGNATURE_INFERENCE_PASS_H__

#include <loco.h>

#include <luci/LuciPass.h>

namespace luci
{

/**
 * @brief Pass to infer shape_signature of nodes
 */
class ShapeSignatureInferencePass : public Pass
{
public:
  const char *name(void) const { return "luci::ShapeSignatureInferencePass"; }

public:
  bool run(luci::Module *module);
  bool run(loco::Graph *graph);
};

} // namespace luci

#endif //__LUCI_SHAPE_SIGNATURE_INFERENCE_PASS_H__
