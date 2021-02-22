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

#include "Canonicalizer.h"

#include "Knob.h"
#include "ProgressReporter.h"

#include "Transforms/ShapeInferencePass.h"
#include "Transforms/TypeInferencePass.h"

#include "Canonicalization/AddCanonicalizer.h"
#include "Canonicalization/AvgPoolCanonicalizer.h"
#include "Canonicalization/BiasAddCanonicalizer.h"
#include "Canonicalization/ConcatV2Canonicalizer.h"
#include "Canonicalization/ConstCanonicalizer.h"
#include "Canonicalization/Conv2DBackpropInputCanonicalizer.h"
#include "Canonicalization/Conv2DCanonicalizer.h"
#include "Canonicalization/DepthwiseConv2dNativeCanonicalizer.h"
#include "Canonicalization/IdentityCanonicalizer.h"
#include "Canonicalization/MaximumCanonicalizer.h"
#include "Canonicalization/MaxPoolCanonicalizer.h"
#include "Canonicalization/MeanCanonicalizer.h"
#include "Canonicalization/MulCanonicalizer.h"
#include "Canonicalization/PadCanonicalizer.h"
#include "Canonicalization/PlaceholderCanonicalizer.h"
#include "Canonicalization/RealDivCanonicalizer.h"
#include "Canonicalization/ReluCanonicalizer.h"
#include "Canonicalization/Relu6Canonicalizer.h"
#include "Canonicalization/ReshapeCanonicalizer.h"
#include "Canonicalization/RsqrtCanonicalizer.h"
#include "Canonicalization/SoftmaxCanonicalizer.h"
#include "Canonicalization/SqrtCanonicalizer.h"
#include "Canonicalization/SqueezeCanonicalizer.h"
#include "Canonicalization/StopGradientCanonicalizer.h"
#include "Canonicalization/SubCanonicalizer.h"
#include "Canonicalization/TanhCanonicalizer.h"
// For virtual nodes
#include "Canonicalization/TFPushCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNodes.h>

#include <logo/Phase.h>

#include <memory>
#include <cassert>

namespace
{

/**
 * @brief Return true if graph has TFDialect nodes
 */
bool has_tf_nodes(loco::Graph *g)
{
  auto active_nodes = loco::active_nodes(loco::output_nodes(g));
  for (auto node : active_nodes)
  {
    if (node->dialect() == moco::TFDialect::get())
    {
      return true;
    }
  }
  return false;
}

} // namespace

namespace moco
{
namespace tf
{

void Canonicalizer::canonicalize(loco::Graph *g) const
{
  logo::Phase phase;

  /* TRANSFORM DECLARATION BEGIN */
  // Run shape and type inference at the top
  phase.emplace_back(std::make_unique<ShapeInferencePass>());
  phase.emplace_back(std::make_unique<TypeInferencePass>());

  phase.emplace_back(std::make_unique<AddCanonicalizer>());
  phase.emplace_back(std::make_unique<AvgPoolCanonicalizer>());
  if (moco::tf::get<moco::tf::Knob::CanonicalizeBiasAdd>())
    phase.emplace_back(std::make_unique<BiasAddCanonicalizer>());
  phase.emplace_back(std::make_unique<ConcatV2Canonicalizer>());
  if (moco::tf::get<moco::tf::Knob::CanonicalizeConst>())
    phase.emplace_back(std::make_unique<ConstCanonicalizer>());
  phase.emplace_back(std::make_unique<Conv2DBackpropInputCanonicalizer>());
  if (moco::tf::get<moco::tf::Knob::CanonicalizeConv2D>())
    phase.emplace_back(std::make_unique<Conv2DCanonicalizer>());
  phase.emplace_back(std::make_unique<DepthwiseConv2dNativeCanonicalizer>());
  phase.emplace_back(std::make_unique<IdentityCanonicalizer>());
  phase.emplace_back(std::make_unique<MaximumCanonicalizer>());
  phase.emplace_back(std::make_unique<MaxPoolCanonicalizer>());
  phase.emplace_back(std::make_unique<MeanCanonicalizer>());
  phase.emplace_back(std::make_unique<MulCanonicalizer>());
  phase.emplace_back(std::make_unique<PadCanonicalizer>());
  phase.emplace_back(std::make_unique<PlaceholderCanonicalizer>());
  phase.emplace_back(std::make_unique<RealDivCanonicalizer>());
  phase.emplace_back(std::make_unique<ReluCanonicalizer>());
  phase.emplace_back(std::make_unique<Relu6Canonicalizer>());
  phase.emplace_back(std::make_unique<ReshapeCanonicalizer>());
  phase.emplace_back(std::make_unique<RsqrtCanonicalizer>());
  phase.emplace_back(std::make_unique<SoftmaxCanonicalizer>());
  phase.emplace_back(std::make_unique<SqrtCanonicalizer>());
  // NOTE SquaredDifference is handled in ResolveSquaredDifference
  phase.emplace_back(std::make_unique<SqueezeCanonicalizer>());
  phase.emplace_back(std::make_unique<StopGradientCanonicalizer>());
  phase.emplace_back(std::make_unique<SubCanonicalizer>());
  phase.emplace_back(std::make_unique<TanhCanonicalizer>());
  // For virtual nodes
  phase.emplace_back(std::make_unique<TFPushCanonicalizer>());
  /* TRANSFORM DECLARATION END */

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);

  // Assert if graph has TF dialect nodes
  assert(!has_tf_nodes(g));
}

} // namespace tf
} // namespace moco
