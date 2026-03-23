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

#include "moco/Import/GraphBuilderRegistry.h"
#include "moco/Import/Nodes.h"

#include <memory>

namespace moco
{

GraphBuilderRegistry::GraphBuilderRegistry()
{
  add("Add", std::make_unique<AddGraphBuilder>());
  add("AvgPool", std::make_unique<AvgPoolGraphBuilder>());
  add("BiasAdd", std::make_unique<BiasAddGraphBuilder>());
  add("ConcatV2", std::make_unique<ConcatV2GraphBuilder>());
  add("Const", std::make_unique<ConstGraphBuilder>());
  add("Conv2D", std::make_unique<Conv2DGraphBuilder>());
  add("Conv2DBackpropInput", std::make_unique<Conv2DBackpropInputGraphBuilder>());
  add("DepthwiseConv2dNative", std::make_unique<DepthwiseConv2dNativeGraphBuilder>());
  add("FakeQuantWithMinMaxVars", std::make_unique<FakeQuantWithMinMaxVarsGraphBuilder>());
  add("FusedBatchNorm", std::make_unique<FusedBatchNormGraphBuilder>());
  add("Identity", std::make_unique<IdentityGraphBuilder>());
  add("Maximum", std::make_unique<MaximumGraphBuilder>());
  add("MaxPool", std::make_unique<MaxPoolGraphBuilder>());
  add("Mean", std::make_unique<MeanGraphBuilder>());
  add("Mul", std::make_unique<MulGraphBuilder>());
  add("Pack", std::make_unique<PackGraphBuilder>());
  add("Pad", std::make_unique<PadGraphBuilder>());
  add("Placeholder", std::make_unique<PlaceholderGraphBuilder>());
  add("RealDiv", std::make_unique<RealDivGraphBuilder>());
  add("Relu", std::make_unique<ReluGraphBuilder>());
  add("Relu6", std::make_unique<Relu6GraphBuilder>());
  add("Reshape", std::make_unique<ReshapeGraphBuilder>());
  add("Rsqrt", std::make_unique<RsqrtGraphBuilder>());
  add("Shape", std::make_unique<ShapeGraphBuilder>());
  add("Softmax", std::make_unique<SoftmaxGraphBuilder>());
  add("Sqrt", std::make_unique<SqrtGraphBuilder>());
  add("SquaredDifference", std::make_unique<SquaredDifferenceGraphBuilder>());
  add("Squeeze", std::make_unique<SqueezeGraphBuilder>());
  add("StopGradient", std::make_unique<StopGradientGraphBuilder>());
  add("StridedSlice", std::make_unique<StridedSliceGraphBuilder>());
  add("Sub", std::make_unique<SubGraphBuilder>());
  add("Tanh", std::make_unique<TanhGraphBuilder>());

  // Virtual node like `TFPush` need not to be added here
}

} // namespace moco
