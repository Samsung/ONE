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

#include "GraphBuilderRegistry.h"

#include "Layer/Concatenation.h"
#include "Layer/Convolution.h"
#include "Layer/Eltwise.h"
#include "Layer/Input.h"
#include "Layer/Pooling.h"
#include "Layer/ReLU.h"
#include "Layer/Scale.h"
#include "Layer/BatchNorm.h"

#include <memory>

using std::make_unique;

namespace caffeimport
{

GraphBuilderRegistry::GraphBuilderRegistry()
{
  _builder_map["Concat"] = make_unique<ConcatBuilder>();
  _builder_map["Convolution"] = make_unique<ConvolutionBuilder>();
  _builder_map["Eltwise"] = make_unique<EltwiseBuilder>();
  _builder_map["Input"] = make_unique<InputBuilder>();
  _builder_map["Pooling"] = make_unique<PoolingBuilder>();
  _builder_map["ReLU"] = make_unique<ReLUBuilder>();
  _builder_map["Scale"] = make_unique<ScaleBuilder>();
  _builder_map["BatchNorm"] = make_unique<BatchNormBuilder>();
}

} // namespace caffeimport
