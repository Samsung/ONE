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

#include <dlfcn.h>

#include "mir/Graph.h"
#include "pass/Pass.h"
#include "pass/PassData.h"
#include "pass/PassException.h"

#include "gtest/gtest.h"

using namespace nnc;

class DummyPass1 : public Pass
{
public:
  PassData run(PassData data) override
  {
    auto graph = static_cast<mir::Graph *>(data);

    if (!graph)
    {
      throw PassException();
    }

    return graph;
  }
};

class DummyPass2 : public Pass
{
public:
  PassData run(PassData data) override
  {
    auto tv = static_cast<mir::TensorVariant *>(data);

    if (!tv)
    {
      throw PassException();
    }

    return nullptr;
  }
};

TEST(CONTRIB_PASS, PassManager)
{
  DummyPass1 pass1;
  DummyPass2 pass2;

  mir::Graph g;
  auto res = pass1.run(&g);
  ASSERT_NE(static_cast<mir::Graph *>(res), nullptr);

  ASSERT_THROW(pass2.run(res), PassException);
}
