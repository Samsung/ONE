/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_GRAPH_VERIFIER_VERIFIER_H__
#define __ONERT_GRAPH_VERIFIER_VERIFIER_H__

namespace onert
{
namespace ir
{
class Graph;
} // namespace ir
} // namespace onert

namespace onert
{
namespace ir
{
namespace verifier
{

struct IVerifier
{
  virtual ~IVerifier() = default;
  virtual bool verify(const Graph &graph) const noexcept = 0;
};

} // namespace verifier
} // namespace ir
} // namespace onert

namespace onert
{
namespace ir
{
namespace verifier
{

class DAGChecker : public IVerifier
{
public:
  bool verify(const Graph &graph) const noexcept override;
};

class EdgeConsistencyChecker : public IVerifier
{
public:
  bool verify(const Graph &graph) const noexcept override;
};

/**
 * @brief Check model input and output operands are really exist in the graph
 */
class InputOutputChecker : public IVerifier
{
public:
  bool verify(const Graph &graph) const noexcept override;
};

} // namespace verifier
} // namespace ir
} // namespace onert

#endif // __ONERT_GRAPH_VERIFIER_VERIFIER_H__
