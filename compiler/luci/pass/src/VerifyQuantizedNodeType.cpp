/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "VerifyQuantizedNodeType.h"

#include <luci/IR/CircleNodes.h>

#include <memory>

namespace luci
{

std::shared_ptr<VerifyQuantizedNodeType> VerifyQuantizedNodeType::create(loco::DataType dtype)
{
  if (dtype == loco::DataType::U8)
    return std::make_shared<VerifyQuantizedNodeU8Type>();
  else if (dtype == loco::DataType::S16)
    return std::make_shared<VerifyQuantizedNodeS16Type>();
  else
    throw std::domain_error("Not supported Quantized type");
}

} // namespace luci
