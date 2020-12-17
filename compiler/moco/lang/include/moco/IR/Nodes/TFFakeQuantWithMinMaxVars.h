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

#ifndef __MOCO_IR_TFFAKEQUANTWITHMINMAXVARS_H__
#define __MOCO_IR_TFFAKEQUANTWITHMINMAXVARS_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

class TFFakeQuantWithMinMaxVars final
  : public FixedArityNode<3, TFNodeImpl<TFOpcode::FakeQuantWithMinMaxVars>>
{
public:
  loco::Node *inputs(void) const { return at(0)->node(); }
  void inputs(Node *node) { at(0)->node(node); }

  loco::Node *min(void) const { return at(1)->node(); }
  void min(Node *node) { at(1)->node(node); }

  loco::Node *max(void) const { return at(2)->node(); }
  void max(Node *node) { at(2)->node(node); }

public:
  const int64_t &num_bits(void) const { return _num_bits; }
  void num_bits(const int64_t &num_bits) { _num_bits = num_bits; }

  const bool &narrow_range(void) const { return _narrow_range; }
  void narrow_range(const bool &narrow_range) { _narrow_range = narrow_range; }

private:
  int64_t _num_bits{8};
  bool _narrow_range{false};
};

} // namespace moco

#endif // __MOCO_IR_TFFAKEQUANTWITHMINMAXVARS_H__
