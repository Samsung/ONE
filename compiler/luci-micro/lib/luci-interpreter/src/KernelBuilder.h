/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELBUILDER_H
#define LUCI_INTERPRETER_KERNELBUILDER_H

#include "TensorMap.h"
#include "core/Kernel.h"

#include <luci/IR/CircleNodeVisitor.h>

#include <memory>

namespace luci_interpreter
{

  class KernelBuilder : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>
  {
  public:
    explicit KernelBuilder(TensorMap &tensor_map) : _tensor_map(tensor_map) {}

    std::unique_ptr<Kernel> visit(const luci::CircleAdd *node) override;
    // std::unique_ptr<Kernel> visit(const luci::CircleAveragePool2D *node) override;
    // std::unique_ptr<Kernel> visit(const luci::CircleConcatenation *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleConv2D *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleConst *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleDepthwiseConv2D *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleFullyConnected *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleInput *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleMaxPool2D *node) override;
    // std::unique_ptr<Kernel> visit(const luci::CircleMul *node) override;
    std::unique_ptr<Kernel> visit(const luci::CircleOutput *node) override;
    // std::unique_ptr<Kernel> visit(const luci::CircleReshape *node) override;
    // std::unique_ptr<Kernel> visit(const luci::CircleSoftmax *node) override;

  private:
    const Tensor *getInputTensor(const loco::Node *node) const
    {
      const Tensor *tensor = _tensor_map.getTensor(node);
      assert(tensor != nullptr);
      return tensor;
    }

    const Tensor *getOptionalInputTensor(const loco::Node *node) const
    {
      // TODO Revise this when optional inputs are implemented in the IR.
      return getInputTensor(node);
    }

    Tensor *getOutputTensor(const loco::Node *node) const
    {
      Tensor *tensor = _tensor_map.getTensor(node);
      assert(tensor != nullptr);
      return tensor;
    }

  private:
    TensorMap &_tensor_map;
  };

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELBUILDER_H
