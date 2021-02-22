/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exo/TFLExporter.h"

#include "TFLExporterImpl.h"

#include <oops/InternalExn.h>

#include <memory>
#include <fstream>

namespace exo
{

TFLExporter::TFLExporter(loco::Graph *graph) : _impl(std::make_unique<Impl>(graph))
{
  // NOTHING TO DO
}

TFLExporter::~TFLExporter() = default;

void TFLExporter::dumpToFile(const char *path) const
{
  const char *ptr = _impl->getBufferPointer();
  const size_t size = _impl->getBufferSize();

  if (!ptr)
    INTERNAL_EXN("Graph was not serialized by FlatBuffer for some reason");

  std::ofstream file(path, std::ofstream::binary);
  file.write(ptr, size);
}

} // namespace exo
