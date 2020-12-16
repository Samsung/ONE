/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RankLayer.h"

#include "OperationUtils.h"

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

RankLayer::RankLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void RankLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void RankLayer::run()
{
  int32_t *output_data = reinterpret_cast<int32_t *>(_output->buffer());
  output_data[0] = _input->getShape().rank();
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
