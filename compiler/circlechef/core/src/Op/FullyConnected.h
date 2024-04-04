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

#ifndef __OP_FULLYCONNECTED_H__
#define __OP_FULLYCONNECTED_H__

#include "OpChef.h"

class FullyConnectedChef final : public OpChef
{
public:
  explicit FullyConnectedChef(const circlechef::Operation *operation) : _operation{operation}
  {
    // DO NOTHING
  }

public:
  circle::BuiltinOperator code(void) const override
  {
    return circle::BuiltinOperator_FULLY_CONNECTED;
  }

  circle::BuiltinOptions type(void) const override
  {
    return circle::BuiltinOptions_FullyConnectedOptions;
  }

  flatbuffers::Offset<void> value(flatbuffers::FlatBufferBuilder &fbb) const override;

private:
  const circlechef::Operation *_operation;
};

struct FullyConnectedChefFactory final : public OpChefFactory
{
  std::unique_ptr<OpChef> create(const circlechef::Operation *operation) const override;
};

#endif // __OP_FULLYCONNECTED_H__
