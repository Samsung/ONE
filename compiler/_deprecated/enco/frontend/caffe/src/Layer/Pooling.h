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

#ifndef __POOLING_BUILDER_H__
#define __POOLING_BUILDER_H__

#include "GraphBuilder.h"

#include "Context.h"

namespace caffeimport
{

class PoolingBuilder final : public GraphBuilder
{
public:
  void build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const override;
};

} // namespace caffeimport

#endif // __POOLING_BUILDER_H__
