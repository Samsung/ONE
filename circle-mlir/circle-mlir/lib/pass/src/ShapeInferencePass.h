/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/core/transforms/shape_inference/pass.h

#ifndef __CIRCLE_MLIR_PASS_SHAPE_INFERENCE_PASS_H__
#define __CIRCLE_MLIR_PASS_SHAPE_INFERENCE_PASS_H__

#include <mlir/Pass/Pass.h>

namespace mlir
{
namespace Circle
{

std::unique_ptr<mlir::Pass> CreateShapeInferencePass(int64_t &dynaCount);
std::unique_ptr<mlir::Pass> CreateShapeValidatePass(void);
std::unique_ptr<mlir::Pass> CreateDynaShapeValidatePass(void);

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_SHAPE_INFERENCE_PASS_H__
