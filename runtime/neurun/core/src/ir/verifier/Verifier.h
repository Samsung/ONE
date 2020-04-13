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

#ifndef __NEURUN_GRAPH_VERIFIER_VERIFIER_H__
#define __NEURUN_GRAPH_VERIFIER_VERIFIER_H__

namespace neurun
{
namespace ir
{
class Graph;
} // namespace ir
} // namespace neurun

namespace neurun
{
namespace ir
{
namespace verifier
{

struct IVerifier
{
  virtual ~IVerifier() = default;
  virtual bool verify(const Graph &graph) const = 0;
};

} // namespace verifier
} // namespace ir
} // namespace neurun

namespace neurun
{
namespace ir
{
namespace verifier
{

class DAGChecker : public IVerifier
{
public:
  bool verify(const Graph &graph) const override;
};

class EdgeConsistencyChecker : public IVerifier
{
public:
  bool verify(const Graph &graph) const override;
};

} // namespace verifier
} // namespace ir
} // namespace neurun

#endif // __NEURUN_GRAPH_VERIFIER_VERIFIER_H__
