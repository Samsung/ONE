/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_IR_TFFUSEDBATCHNORM_H__
#define __MOCO_IR_TFFUSEDBATCHNORM_H__

#include "moco/IR/TFNodeDecl.h"

namespace moco
{

class TFFusedBatchNorm final : public FixedArityNode<5, TFNodeImpl<TFOpcode::FusedBatchNorm>>
{
public:
  TFFusedBatchNorm() = default;

public:
  Node *x(void) const { return at(0)->node(); }
  void x(Node *node) { at(0)->node(node); }

  Node *scale(void) const { return at(1)->node(); } // gamma
  void scale(Node *node) { at(1)->node(node); }

  Node *offset(void) const { return at(2)->node(); } // beta
  void offset(Node *node) { at(2)->node(node); }

  Node *mean(void) const { return at(3)->node(); }
  void mean(Node *node) { at(3)->node(node); }

  Node *variance(void) const { return at(4)->node(); }
  void variance(Node *node) { at(4)->node(node); }

  float epsilon(void) const { return _epsilon; }
  void epsilon(float epsilon) { _epsilon = epsilon; }

private:
  float _epsilon = 0.001f;
};

} // namespace moco

#endif // __MOCO_IR_TFFUSEDBATCHNORM_H__
