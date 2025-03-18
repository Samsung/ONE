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

#ifndef __CIRCLE_MLIR_PASS_CIRCLE_PASS_H__
#define __CIRCLE_MLIR_PASS_CIRCLE_PASS_H__

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

namespace mlir
{
namespace Circle
{

int preprocessONNX(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int shapeInferenceONNX(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int convertToCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int postProcessCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int shapeValidateCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int dynaShapeValidateCircle(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module);
int dumpCircleOps(llvm::raw_fd_ostream &os, mlir::MLIRContext &context,
                  mlir::OwningOpRef<mlir::ModuleOp> &module);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_CIRCLE_PASS_H__
