/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORT_OP_CIRCLE_MAXPOOL2D_GRAD_H__
#define __LUCI_IMPORT_OP_CIRCLE_MAXPOOL2D_GRAD_H__

#include "luci/Import/GraphBuilder.h"

namespace luci
{

class CircleMaxPool2DGradGraphBuilder : public GraphBuilder
{
public:
  bool validate(const ValidateArgs &args) const final;

private:
  CircleNode *build_node(const circle::OperatorT &op, const std::vector<CircleNode *> &inputs,
                         loco::Graph *graph) const final;
};

} // namespace luci

#endif // __LUCI_IMPORT_OP_CIRCLE_MAXPOOL2D_GRAD_H__
