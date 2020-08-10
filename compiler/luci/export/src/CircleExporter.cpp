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

#include "luci/CircleExporter.h"
#include "luci/IR/Module.h"
#include "CircleExporterImpl.h"

#include <oops/InternalExn.h>

#include <fstream>
#include <memory>

namespace luci
{

// TODO remove this
Module *CircleExporter::Contract::module(void) const { return nullptr; }

CircleExporter::CircleExporter()
{
  // NOTHING TO DO
}

bool CircleExporter::invoke(Contract *contract) const
{
  auto module = contract->module();
  if (module != nullptr)
  {
    CircleExporterImpl impl(module);

    const char *ptr = impl.getBufferPointer();
    const size_t size = impl.getBufferSize();

    // we just send one time
    return contract->store(ptr, size);
  }

  auto graph = contract->graph();
  if (graph == nullptr)
    return false;

  CircleExporterImpl impl(graph);

  const char *ptr = impl.getBufferPointer();
  const size_t size = impl.getBufferSize();

  // we just send one time
  return contract->store(ptr, size);
}

} // namespace luci
