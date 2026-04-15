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

#include "ANN/Context.h"

#include <memory>

ANNBinder *ANNContext::create(coco::Block *blk)
{
  auto mod = std::make_unique<ann::Module>();
  auto obj = std::make_unique<ANNBinder>(blk, std::move(mod));
  auto ptr = obj.get();

  _binders.emplace_back(std::move(obj));
  _map[blk] = ptr;

  return ptr;
}
