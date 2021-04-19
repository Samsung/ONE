/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CONNECT_NODE_TEST_H__
#define __CONNECT_NODE_TEST_H__

#include "ConnectNode.h"

#include <luci/test/TestIOGraph.h>

#include <loco/IR/Graph.h>

#include <initializer_list>
#include <stdexcept>

namespace luci
{
namespace test
{

template <unsigned N> class TestIsOGraph : public TestIsGraphlet<N>, public TestOGraphlet
{
public:
  TestIsOGraph() = default;

public:
  virtual void init(const std::initializer_list<ShapeU32> shape_in, const ShapeU32 shape_out)
  {
    if (shape_in.size() != N)
      throw std::runtime_error("Failed to init TestIsOGraph");

    TestIsGraphlet<N>::init(TestIsGraphlet<N>::g(), shape_in);
    TestOGraphlet::init(TestIsGraphlet<N>::g(), shape_out);
  }
};

} // namespace test
} // namespace luci

#endif // __CONNECT_NODE_TEST_H__
