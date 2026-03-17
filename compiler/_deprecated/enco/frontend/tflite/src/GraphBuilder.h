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

#ifndef __GRAPH_BUILDER_H__
#define __GRAPH_BUILDER_H__

#include "Context.h"

#include <schema_generated.h>

namespace tflimport
{

/**
 * @brief Parent class of tflite operation graph builders (e.g., Conv2DGraphBuilder)
 */
class GraphBuilder
{
public:
  /**
   * TODO Declare "validate" method as a pure virtual method
   *
   * Q: Is it possible to validate T/F Lite model only with this interface?
   */
  virtual bool validate(const tflite::Operator *) const { return true; }

  virtual void build(const tflite::Operator *op, GraphBuilderContext *context) const = 0;
  virtual ~GraphBuilder() {}
};

} // namespace tflimport

#endif // __GRAPH_BUILDER_H__
