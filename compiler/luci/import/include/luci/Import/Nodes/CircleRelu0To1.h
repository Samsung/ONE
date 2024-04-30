/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORT_OP_CIRCLE_RELU_0_TO_1_H__
#define __LUCI_IMPORT_OP_CIRCLE_RELU_0_TO_1_H__

#include "luci/Import/GraphBuilder.h"

namespace luci
{

class CircleRelu0To1GraphBuilder : public GraphBuilder
{
public:
  bool validate(const ValidateArgs &args) const final;

private:
  CircleNode *build_node(const circle::OperatorT &op, const std::vector<CircleNode *> &inputs,
                         loco::Graph *graph) const final;
};

} // namespace luci

#endif // __LUCI_IMPORT_OP_CIRCLE_RELU_0_TO_1_H__
