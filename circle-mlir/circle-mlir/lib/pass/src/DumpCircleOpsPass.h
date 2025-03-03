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

#ifndef __CIRCLE_MLIR_PASS_DUMP_CIRCLE_OPS_PASS_H__
#define __CIRCLE_MLIR_PASS_DUMP_CIRCLE_OPS_PASS_H__

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>

#include <functional>

namespace mlir
{
namespace Circle
{

struct DumpCircleOpsPass
  : public mlir::PassWrapper<DumpCircleOpsPass, mlir::OperationPass<mlir::func::FuncOp>>
{
  DumpCircleOpsPass() = default;
  DumpCircleOpsPass(const DumpCircleOpsPass &pass)
    : mlir::PassWrapper<DumpCircleOpsPass, OperationPass<mlir::func::FuncOp>>()
  {
    _getOStream = pass._getOStream;
  }

  llvm::StringRef getArgument() const override { return "circle-dump-ops"; }

  llvm::StringRef getDescription() const override { return "Dump Circle ops"; }

  Option<std::string> target{*this, "target", ::llvm::cl::desc("Dump Circle operators"),
                             ::llvm::cl::init("")};

  void runOnOperation() final;

protected:
  void dumpRegion(mlir::Region &region);

public:
  using GetOStream_t = std::function<llvm::raw_fd_ostream &(void)>;

  void ostream(GetOStream_t os) { _getOStream = os; }
  llvm::raw_fd_ostream &ostream(void) { return _getOStream(); }

protected:
  GetOStream_t _getOStream = nullptr;
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_DUMP_CIRCLE_OPS_PASS_H__
