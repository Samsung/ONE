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

#ifndef __MOCO_IR_TFDEPTHWISECONV2DNATIVE_H__
#define __MOCO_IR_TFDEPTHWISECONV2DNATIVE_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

class TFDepthwiseConv2dNative final
  : public FixedArityNode<2, TFNodeImpl<TFOpcode::DepthwiseConv2dNative>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(Node *node) { at(1)->node(node); }

public:
  const TFPadding &padding(void) const { return _padding; }
  void padding(const TFPadding &padding) { _padding = padding; }

  const TFDataLayout &data_layout(void) const { return _data_layout; }
  void data_layout(const TFDataLayout &data_layout) { _data_layout = data_layout; }

  const std::vector<int64_t> &strides(void) const { return _strides; }
  void strides(const std::vector<int64_t> &strides) { _strides = strides; }

private:
  TFPadding _padding;
  TFDataLayout _data_layout;
  std::vector<int64_t> _strides;
  // TODO Support "Dilation"
};

} // namespace moco

#endif // __MOCO_IR_TFDEPTHWISECONV2DNATIVE_H__
