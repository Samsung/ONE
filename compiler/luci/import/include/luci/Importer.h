/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORTER_H__
#define __LUCI_IMPORTER_H__

#include "luci/Import/GraphBuilderRegistry.h"

#include "luci/IR/Module.h"

#include <loco.h>

#include <mio/circle/schema_generated.h>

#include <memory>

namespace luci
{

class Importer final
{
public:
  Importer();

public:
  explicit Importer(const GraphBuilderSource *source) : _source{source}
  {
    // DO NOTHING
  }

public:
  std::unique_ptr<loco::Graph> import(const circle::Model *model) const;
  std::unique_ptr<Module> importModule(const circle::Model *model) const;

private:
  const GraphBuilderSource *_source = nullptr;
};

} // namespace luci

#endif // __LUCI_IMPORTER_H__
