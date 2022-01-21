/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_OPERATION_EXPORTER_RULE_H__
#define __CIRCLE_OPERATION_EXPORTER_RULE_H__

#include "CircleOperationExporter.h"

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

struct ExportContext
{
  flatbuffers::FlatBufferBuilder &builder;
  luci::SerializedModelData &md;
  luci::SerializedGraphData &gd;
};

class OperationExporterRule final : public luci::CircleNodeMutableVisitor<void>
{
public:
  OperationExporterRule(ExportContext &ctx) : _ctx{ctx}
  {
    // DO NOTHING
  }

public:
  // Default export rule
  void visit(luci::CircleNode *node) final;

  // Non-virtual
  void visit(luci::CircleConst *) final{/* skip, everything is done in exportOpDefinedTensors */};

  // Virtual
  void visit(luci::CircleInput *) final {}
  void visit(luci::CircleOutput *) final {}
  void visit(luci::CircleOutputDummy *) final {}
  void visit(luci::CircleOutputExclude *) final {}
  // Virtual for multiple-outputs
  void visit(luci::CircleBidirectionalSequenceLSTMOut *) final {}
  void visit(luci::CircleCustomOut *) final {}
  void visit(luci::CircleIfOut *) final {}
  void visit(luci::CircleNonMaxSuppressionV4Out *) final {}
  void visit(luci::CircleNonMaxSuppressionV5Out *) final {}
  void visit(luci::CircleSplitOut *) final {}
  void visit(luci::CircleSplitVOut *) final {}
  void visit(luci::CircleTopKV2Out *) final {}
  void visit(luci::CircleUniqueOut *) final {}
  void visit(luci::CircleUnpackOut *) final {}
  void visit(luci::CircleVariable *) final {}
  void visit(luci::CircleWhileOut *) final {}

protected:
  ExportContext &_ctx;
};

} // namespace luci

#endif // __CIRCLE_OPERATION_EXPORTER_RULE_H__
