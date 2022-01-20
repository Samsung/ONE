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
#include "luci_interpreter/GraphBuilderRegistry.h"
#include "Nodes/CircleReferencingConst.h"

namespace luci_interpreter
{

std::unique_ptr<luci::GraphBuilderSource> source_without_constant_copying()
{
  auto builder = std::make_unique<luci::GraphBuilderRegistry>();
  {
    // redefine NodeBuilder of BUFFER type
    builder->add(std::make_unique<CircleReferencingConstNodeBuilder>());
  }

  return builder;
}

} // namespace luci_interpreter
