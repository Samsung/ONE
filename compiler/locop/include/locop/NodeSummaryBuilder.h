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

#ifndef __LOCOP_NODE_SUMMARY_BUILDER_H__
#define __LOCOP_NODE_SUMMARY_BUILDER_H__

#include "locop/SymbolTable.h"
#include "locop/NodeSummary.h"

#include <loco.h>

namespace locop
{

/**
 * @brief Build a summary from loco Node
 */
struct NodeSummaryBuilder
{
  virtual ~NodeSummaryBuilder() = default;

  virtual bool build(const loco::Node *, NodeSummary &) const = 0;
};

struct NodeSummaryBuilderFactory
{
  virtual ~NodeSummaryBuilderFactory() = default;

  virtual std::unique_ptr<NodeSummaryBuilder> create(const SymbolTable *) const = 0;
};

} // namespace locop

#endif // __LOCOP_NODE_SUMMARY_BUILDER_H__
