/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "InferenceContext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "Buffer.h"
#include "ClDevice.h"

#include "kernels/GpuOperation.h"
#include "ModelHints.h"
#include "Precision.h"
#include "StorageTypeUtil.h"
#include "TensorType.h"
#include "DataType.h"
#include "Model.h"
#include "Operations.h"
#include "Shape.h"
#include "Types.h"
#include "Util.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

CLNode::CLNode(CLNode &&node)
  : operation(std::move(node.operation)), inputs(std::move(node.inputs)),
    outputs(std::move(node.outputs)), name(std::move(node.name))
{
}

CLNode &CLNode::operator=(CLNode &&node)
{
  if (this != &node)
  {
    operation = std::move(node.operation);
    inputs = std::move(node.inputs);
    outputs = std::move(node.outputs);
    name = std::move(node.name);
  }
  return *this;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
