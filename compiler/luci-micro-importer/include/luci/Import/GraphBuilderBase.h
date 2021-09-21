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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_BASE_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_BASE_H__

#include "GraphBuilderContext.h"

#include <luci/IR/CircleNode.h>

#include <mio/circle/schema_generated.h>

namespace luci
{

/**
 * @brief Interface of convert circle::OperatorT to CircleNode
 */
struct GraphBuilderBase
{
  struct ValidateArgs
  {
    ValidateArgs(const circle::OperatorT &o, const CircleReader &r) : op(o), reader(r) {}

    const circle::OperatorT &op;
    const CircleReader &reader;
  };

  virtual bool validate(const ValidateArgs &) const = 0;
  virtual CircleNode *build(const circle::OperatorT &op, GraphBuilderContext *context) const = 0;

  virtual ~GraphBuilderBase() = default;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_BASE_H__
