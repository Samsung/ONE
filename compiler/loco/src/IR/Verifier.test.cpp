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

#include "loco/IR/Verifier.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using std::make_unique;

TEST(VerifierTest, valid_minimal)
{
  auto g = loco::make_graph();
  auto push = g->nodes()->create<loco::Push>();

  ASSERT_FALSE(loco::valid(g.get()));
}

TEST(VerifierTest, valid_error_reporter)
{
  using namespace loco;

  auto g = loco::make_graph();
  auto push = g->nodes()->create<loco::Push>();

  class Collector final : public loco::ErrorListener
  {
  public:
    Collector(std::vector<ErrorDetail<ErrorCategory::MissingArgument>> *out) : _out{out}
    {
      // DO NOTHING
    }

  public:
    void notify(const ErrorDetail<ErrorCategory::MissingArgument> &d) override
    {
      _out->emplace_back(d);
    }

  private:
    std::vector<ErrorDetail<ErrorCategory::MissingArgument>> *_out;
  };

  std::vector<ErrorDetail<ErrorCategory::MissingArgument>> errors;
  ASSERT_FALSE(loco::valid(g.get(), make_unique<Collector>(&errors)));
  ASSERT_EQ(1, errors.size());
  ASSERT_EQ(push, errors.at(0).node());
  ASSERT_EQ(0, errors.at(0).index());
}
