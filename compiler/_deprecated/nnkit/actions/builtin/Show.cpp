/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nnkit/Action.h>

#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <iostream>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;

std::ostream &operator<<(std::ostream &os, const Index &index)
{
  if (index.rank() > 0)
  {
    os << index.at(0);
    for (uint32_t axis = 1; axis < index.rank(); ++axis)
    {
      os << "," << index.at(axis);
    }
  }
  return os;
}

struct ShowAction final : public nnkit::Action
{
  void run(nnkit::TensorContext &ctx) override;
};

void ShowAction::run(nnkit::TensorContext &ctx)
{
  std::cout << "count: " << ctx.size() << std::endl;
  for (uint32_t n = 0; n < ctx.size(); ++n)
  {
    std::cout << "  tensor(" << n << ") : " << ctx.name(n) << std::endl;

    using nncc::core::ADT::tensor::Reader;
    using nnkit::TensorContext;

    ctx.getConstFloatTensor(n, [](const TensorContext &ctx, uint32_t n, const Reader<float> &t) {
      for (IndexEnumerator e{ctx.shape(n)}; e.valid(); e.advance())
      {
        const auto &index = e.current();

        std::cout << "    " << index << ": " << t.at(index) << std::endl;
      }
    });
  }
}

#include <nnkit/CmdlineArguments.h>

#include <memory>

extern "C" std::unique_ptr<nnkit::Action> make_action(const nnkit::CmdlineArguments &args)
{
  return std::make_unique<ShowAction>();
}
