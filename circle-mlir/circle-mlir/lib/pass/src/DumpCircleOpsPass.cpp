/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DumpCircleOpsPass.h"

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir
{
namespace Circle
{

void DumpCircleOpsPass::runOnOperation()
{
  mlir::func::FuncOp func = getOperation();

  for (auto &region : func->getRegions())
    dumpRegion(region);
}

void DumpCircleOpsPass::dumpRegion(mlir::Region &region)
{
  region.walk([&](mlir::Operation *op) { ostream() << op->getName() << "\n"; });

  region.walk([&](mlir::Operation *op) {
    for (auto &region : op->getRegions())
      dumpRegion(region);
  });
}

} // namespace Circle
} // namespace mlir
