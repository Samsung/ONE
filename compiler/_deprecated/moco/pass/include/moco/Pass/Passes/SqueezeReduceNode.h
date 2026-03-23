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

#ifndef __MOCO_PASS_SQUEEZE_REDUCE_NODE_H__
#define __MOCO_PASS_SQUEEZE_REDUCE_NODE_H__

#include <logo/Pass.h>

#include <loco.h>

namespace moco
{

/**
 * @brief  If ReduceTypeOP don't keep dimensions, replace the ReduceTypeOp
 *         as new one to keep dimensions and insert TFSqueeze
 */
class SqueezeReduceNode : public logo::Pass
{
public:
  const char *name(void) const final { return "SqueezeReduceNode"; }

public:
  bool run(loco::Graph *graph) override;
};

} // namespace moco

#endif // __MOCO_PASS_SQUEEZE_REDUCE_NODE_H__
