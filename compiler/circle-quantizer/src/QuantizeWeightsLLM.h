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

#ifndef LUCI_QUANTIZE_WEIGHTS_LLM_H
#define LUCI_QUANTIZE_WEIGHTS_LLM_H

#include <luci/IR/CircleNodeVisitor.h>

namespace quantizer
{

class QuantizeWeightsLLM : public luci::CircleNodeMutableVisitor<void>
{
public:
  enum Type
  {
    Q4_0,
    Q8_0,
    SKIP // hidden type for gather indice type change to int32
  };

public:
  QuantizeWeightsLLM(Type type, int32_t skip_length) : _quant_type(type), _skip_length(skip_length)
  {
  }

private:
  void visit(luci::CircleFullyConnected *node);
  void visit(luci::CircleGather *node);
  void visit(luci::CircleNode *);

private:
  Type _quant_type;
  int32_t _skip_length;
};

} // namespace quantizer

#endif // LUCI_QUANTIZE_WEIGHTS_LLM_H
