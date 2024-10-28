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

#ifndef __RECORD_HESSIAN_HESSIANCOMPUTER_H__
#define __RECORD_HESSIAN_HESSIANCOMPUTER_H__

#include "record-hessian/HessianVector.h"

#include <luci/IR/CircleNode.h>
#include <luci_interpreter/Interpreter.h>

namespace record_hessian
{

class HessianComputer
{
public:
  // Record min/max of node
  void recordHessian(const luci::CircleNode *node, const luci_interpreter::Tensor *input_tensor);

  void recordHessianForConv2D(const luci::CircleNode *node, const luci_interpreter::Tensor *input_tensor);
  
  void recordHessianForFullyConnected(const luci::CircleNode *node, const luci_interpreter::Tensor *input_tensor);

  void unfold(std::vector<float> &buf, uint32_t input_n, uint32_t input_h, uint32_t input_w,
              uint32_t input_c, uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h,
              uint32_t dilation_w, uint32_t kernel_oc, uint32_t kernel_h, uint32_t kernel_w,
              uint32_t kernel_ic);

  std::unique_ptr<HessianMap> getMap() const
  {

    auto hessian_map = std::make_unique<HessianMap>();

    for (auto item : _hessian_map)
    {
      auto &vec = (*hessian_map)[item.first];
      vec = item.second.hessian;
    }

    return hessian_map;
  }

private:
  std::unordered_map<const luci::CircleNode *, HessianVector> _hessian_map;
};
} // namespace record_hessian

#endif // __RECORD_HESSIAN_HESSIANCOMPUTER_H__
