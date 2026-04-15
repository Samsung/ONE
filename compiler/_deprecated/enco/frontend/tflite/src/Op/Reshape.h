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

#ifndef __OP_RESHAPE_H__
#define __OP_RESHAPE_H__

#include "GraphBuilder.h"

#include <schema_generated.h>

namespace tflimport
{

/**
 * @brief GraphBuilder for Reshape operator
 */
class ReshapeGraphBuilder : public GraphBuilder
{
public:
  void build(const tflite::Operator *op, GraphBuilderContext *) const override;
};

} // namespace tflimport

#endif // __OP_RESHAPE_H__
