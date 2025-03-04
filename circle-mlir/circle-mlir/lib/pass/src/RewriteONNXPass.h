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

#ifndef __CIRCLE_MLIR_PASS_REWRITE_ONNX_PASS_H__
#define __CIRCLE_MLIR_PASS_REWRITE_ONNX_PASS_H__

#include <mlir/Dialect/Func/Transforms/Passes.h>

namespace mlir
{
namespace Circle
{

std::unique_ptr<mlir::Pass> createRewriteONNXPass(void);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_REWRITE_ONNX_PASS_H__
