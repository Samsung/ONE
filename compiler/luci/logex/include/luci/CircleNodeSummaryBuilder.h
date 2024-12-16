/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER__
#define __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER__

#include <luci/IR/CircleNode.h>
#include <locop/NodeSummary.h>
#include <locop/SymbolTable.h>

#include <memory>
#include <sstream>
#include <vector>

namespace luci
{

class CircleNodeSummaryBuilder
{
public:
  bool build(const loco::Node *node, const locop::SymbolTable *tbl, locop::NodeSummary &s);

private:
  /**
   * @brief Template methods for building node summary.
   *        Default behavior is building a node which has no input.
   */
  virtual bool validate(const luci::CircleNode *node);
  virtual std::vector<std::string> get_input_names(const luci::CircleNode *node);
  virtual void build_attributes(const luci::CircleNode *node, locop::NodeSummary &s);
  virtual void update_status(locop::NodeSummary &s);

private:
  std::unique_ptr<CircleNodeSummaryBuilder> create_builder(const luci::CircleNode *node);
};

} // namespace luci

#endif // __LUCI_LOGEX_CIRCLE_NODE_SUMMARY_BUILDER__
