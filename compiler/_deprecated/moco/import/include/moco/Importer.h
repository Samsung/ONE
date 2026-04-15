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

#ifndef __MOCO_IMPORTER_H__
#define __MOCO_IMPORTER_H__

#include "moco/Import/ModelSignature.h"
#include "moco/Import/GraphBuilderRegistry.h"

#include <moco/Names.h>

#include <loco.h>

#include <tensorflow/core/framework/graph.pb.h>

#include <memory>

namespace moco
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
  std::unique_ptr<loco::Graph> import(const ModelSignature &, tensorflow::GraphDef &) const;

private:
  const GraphBuilderSource *_source = nullptr;
};

} // namespace moco

#endif // __MOCO_IMPORTER_H__
