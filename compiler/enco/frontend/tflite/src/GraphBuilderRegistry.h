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

#ifndef __GRAPH_BUILDER_REGISTRY_H__
#define __GRAPH_BUILDER_REGISTRY_H__

#include "Op/Conv2D.h"
#include "Op/DepthwiseConv2D.h"
#include "Op/AveragePool2D.h"
#include "Op/MaxPool2D.h"
#include "Op/Concatenation.h"
#include "Op/ReLU.h"
#include "Op/ReLU6.h"
#include "Op/Reshape.h"
#include "Op/Sub.h"
#include "Op/Div.h"

#include <schema_generated.h>

#include <memory>
#include <map>

using std::make_unique;

namespace tflimport
{

/**
 * @brief Class to return graph builder for passed tflite::builtinOperator
 */
class GraphBuilderRegistry
{
public:
  /**
   * @brief Returns registered GraphBuilder pointer for BuiltinOperator or
   *        nullptr if not registered
   */
  const GraphBuilder *lookup(tflite::BuiltinOperator op) const
  {
    if (_builder_map.find(op) == _builder_map.end())
      return nullptr;

    return _builder_map.at(op).get();
  }

  static GraphBuilderRegistry &get()
  {
    static GraphBuilderRegistry me;
    return me;
  }

private:
  GraphBuilderRegistry()
  {
    // add GraphBuilder for each tflite operation.
    _builder_map[tflite::BuiltinOperator_CONV_2D] = make_unique<Conv2DGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_DEPTHWISE_CONV_2D] =
      make_unique<DepthwiseConv2DGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_AVERAGE_POOL_2D] = make_unique<AvgPool2DGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_MAX_POOL_2D] = make_unique<MaxPool2DGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_CONCATENATION] = make_unique<ConcatenationGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_RELU] = make_unique<ReLUGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_RELU6] = make_unique<ReLU6GraphBuilder>();
    _builder_map[tflite::BuiltinOperator_RESHAPE] = make_unique<ReshapeGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_SUB] = make_unique<SubGraphBuilder>();
    _builder_map[tflite::BuiltinOperator_DIV] = make_unique<DivGraphBuilder>();
  }

private:
  std::map<tflite::BuiltinOperator, std::unique_ptr<GraphBuilder>> _builder_map;
};

} // namespace tflimport

#endif // __GRAPH_BUILDER_REGISTRY_H__
