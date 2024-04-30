/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __OP_RELU_0_TO_1_H__
#define __OP_RELU_0_TO_1_H__

#include "OpChef.h"

class ReLU0To1Chef final : public OpChef
{
public:
  explicit ReLU0To1Chef(const tflchef::Operation *operation) : _operation{operation}
  {
    // DO NOTHING
  }

public:
  tflite::BuiltinOperator code(void) const override { return tflite::BuiltinOperator_RELU_0_TO_1; }

  tflite::BuiltinOptions type(void) const override { return tflite::BuiltinOptions_NONE; }

  flatbuffers::Offset<void> value(flatbuffers::FlatBufferBuilder &fbb) const override;

private:
  const tflchef::Operation *_operation;
};

struct ReLU0To1ChefFactory final : public OpChefFactory
{
  std::unique_ptr<OpChef> create(const tflchef::Operation *operation) const override;
};

#endif // __OP_RELU_0_TO_1_H__
