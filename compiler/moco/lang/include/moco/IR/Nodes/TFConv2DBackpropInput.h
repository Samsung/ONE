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

#ifndef __MOCO_IR_TFCONV2DBACKPROPINPUT_H__
#define __MOCO_IR_TFCONV2DBACKPROPINPUT_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

/// @note TFConv2DBackpropInput corresponds to the following GraphDef
/*
node {
  name: "conv2d_backprop_input"
  op: "Conv2DBackpropInput"
  input: "input_sizes"
  input: "filter"
  input: "out_backprop"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NHWC" }
  }
  attr {
    key: "dilations"
    value {
      list { i: 1 i: 1 i: 1 i: 1 }
    }
  }
  attr {
    key: "padding"
    value { s: "SAME" }
  }
  attr {
    key: "strides"
    value {
      list { i: 1 i: 2 i: 2 i: 1 }
    }
  }
}
*/

/**
 * @note  For Tensorflow Conv2DBackpropInput, 'input' refers actual output of the
 *        node, and 'input' refers actual input. The reasone of this is, as name
 *        suggests, because it is inspired from backpropagation of convolution.
 *        For example, 'out_backprop' of Conv2DBackpropInput is its actual input
 *        feature map, and 'input_sizes' means desired output node's size.
 *        Note that this convention is against loco canonical's convention.
 */
class TFConv2DBackpropInput final
  : public FixedArityNode<3, TFNodeImpl<TFOpcode::Conv2DBackpropInput>>
{
public:
  loco::Node *input_sizes(void) const { return at(0)->node(); }
  void input_sizes(Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(Node *node) { at(1)->node(node); }

  loco::Node *out_backprop(void) const { return at(2)->node(); }
  void out_backprop(Node *node) { at(2)->node(node); }

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

#endif // __MOCO_IR_TFCONV2DBACKPROPINPUT_H__
