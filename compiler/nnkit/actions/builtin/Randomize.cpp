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

#include <chrono>
#include <random>

using nnkit::TensorContext;

struct RandomizeAction final : public nnkit::Action
{
  void run(TensorContext &ctx) override
  {
    int seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::minstd_rand rand(seed);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    for (uint32_t n = 0; n < ctx.size(); ++n)
    {
      using nncc::core::ADT::tensor::Accessor;

      auto fn = [&dist, &rand](const TensorContext &ctx, uint32_t n, Accessor<float> &t) {
        using nncc::core::ADT::tensor::Index;
        using nncc::core::ADT::tensor::IndexEnumerator;

        for (IndexEnumerator e{ctx.shape(n)}; e.valid(); e.advance())
        {
          t.at(e.current()) = dist(rand);
        }
      };

      ctx.getMutableFloatTensor(n, fn);
    }
  }
};

#include <nnkit/CmdlineArguments.h>

#include <memory>

extern "C" std::unique_ptr<nnkit::Action> make_action(const nnkit::CmdlineArguments &args)
{
  return std::make_unique<RandomizeAction>();
}
