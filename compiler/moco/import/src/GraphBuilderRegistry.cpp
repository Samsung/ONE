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

#include <stdex/Memory.h>

namespace moco
{

GraphBuilderRegistry::GraphBuilderRegistry()
{
  add("Add", stdex::make_unique<AddGraphBuilder>());
  add("AvgPool", stdex::make_unique<AvgPoolGraphBuilder>());
  add("BiasAdd", stdex::make_unique<BiasAddGraphBuilder>());
  add("ConcatV2", stdex::make_unique<ConcatV2GraphBuilder>());
  add("Const", stdex::make_unique<ConstGraphBuilder>());
  add("Conv2D", stdex::make_unique<Conv2DGraphBuilder>());
  add("Conv2DBackpropInput", stdex::make_unique<Conv2DBackpropInputGraphBuilder>());
  add("DepthwiseConv2dNative", stdex::make_unique<DepthwiseConv2dNativeGraphBuilder>());
  add("FakeQuantWithMinMaxVars", stdex::make_unique<FakeQuantWithMinMaxVarsGraphBuilder>());
  add("FusedBatchNorm", stdex::make_unique<FusedBatchNormGraphBuilder>());
  add("Identity", stdex::make_unique<IdentityGraphBuilder>());
  add("Maximum", stdex::make_unique<MaximumGraphBuilder>());
  add("MaxPool", stdex::make_unique<MaxPoolGraphBuilder>());
  add("Mean", stdex::make_unique<MeanGraphBuilder>());
  add("Mul", stdex::make_unique<MulGraphBuilder>());
  add("Pack", stdex::make_unique<PackGraphBuilder>());
  add("Pad", stdex::make_unique<PadGraphBuilder>());
  add("Placeholder", stdex::make_unique<PlaceholderGraphBuilder>());
  add("RealDiv", stdex::make_unique<RealDivGraphBuilder>());
  add("Relu", stdex::make_unique<ReluGraphBuilder>());
  add("Relu6", stdex::make_unique<Relu6GraphBuilder>());
  add("Reshape", stdex::make_unique<ReshapeGraphBuilder>());
  add("Rsqrt", stdex::make_unique<RsqrtGraphBuilder>());
  add("Shape", stdex::make_unique<ShapeGraphBuilder>());
  add("Softmax", stdex::make_unique<SoftmaxGraphBuilder>());
  add("Sqrt", stdex::make_unique<SqrtGraphBuilder>());
  add("SquaredDifference", stdex::make_unique<SquaredDifferenceGraphBuilder>());
  add("Squeeze", stdex::make_unique<SqueezeGraphBuilder>());
  add("StopGradient", stdex::make_unique<StopGradientGraphBuilder>());
  add("StridedSlice", stdex::make_unique<StridedSliceGraphBuilder>());
  add("Sub", stdex::make_unique<SubGraphBuilder>());
  add("Tanh", stdex::make_unique<TanhGraphBuilder>());

  // Virtual node like `TFPush` need not to be added here
}

} // namespace moco
