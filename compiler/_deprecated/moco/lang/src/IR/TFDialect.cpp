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

#include "moco/IR/TFDialect.h"
#include "moco/IR/TFNode.h"

#include <loco/IR/Graph.h>
#include <loco/IR/GraphInputIndex.h>
#include <loco/IR/GraphOutputIndex.h>

#include <memory>
#include <cassert>
#include <stdexcept>

namespace
{

struct GiiQueryServiceImpl final : public loco::GraphInputIndexQueryService
{
  bool associated(const loco::Node *node) const final
  {
    if (auto tfplaceholder = dynamic_cast<const moco::TFPlaceholder *>(node))
    {
      return moco::indexed(tfplaceholder);
    }
    return false;
  }

  loco::GraphOutputIndex index(const loco::Node *node) const final
  {
    assert(associated(node));
    auto tfplaceholder = dynamic_cast<const moco::TFPlaceholder *>(node);
    assert(tfplaceholder != nullptr);
    return moco::index(tfplaceholder);
  }
};

struct GoiQueryServiceImpl final : public loco::GraphOutputIndexQueryService
{
  bool associated(const loco::Node *node) const final
  {
    if (auto tfpush = dynamic_cast<const moco::TFPush *>(node))
    {
      return tfpush->indexed();
    }
    return false;
  }

  loco::GraphOutputIndex index(const loco::Node *node) const final
  {
    assert(associated(node));
    if (auto tfpush = dynamic_cast<const moco::TFPush *>(node))
    {
      return tfpush->index();
    }
    throw std::invalid_argument("node");
  }
};

} // namespace

namespace moco
{

TFDialect::TFDialect()
{
  service<loco::GraphInputIndexQueryService>(std::make_unique<GiiQueryServiceImpl>());
  service<loco::GraphOutputIndexQueryService>(std::make_unique<GoiQueryServiceImpl>());
}

loco::Dialect *TFDialect::get(void)
{
  static TFDialect d;
  return &d;
}

} // namespace moco
