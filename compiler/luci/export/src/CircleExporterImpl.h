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

#ifndef __CIRCLE_EXPORTER_IMPL_H__
#define __CIRCLE_EXPORTER_IMPL_H__

#include "luci/CircleExporter.h"
#include "luci/IR/Module.h"

#include "SerializedData.h"

#include <mio/circle/schema_generated.h>

#include <loco.h>

namespace luci
{

/**
 * internal implementation of interface exporter class
 */
class CircleExporterImpl
{
public:
  CircleExporterImpl() = delete;
  ~CircleExporterImpl() = default;

  explicit CircleExporterImpl(loco::Graph *graph);
  explicit CircleExporterImpl(Module *module);

  /**
   * @return pointer to buffer with serialized graph
   */
  const char *getBufferPointer() const;

  /**
   * @return size of buffer with serialized graph
   */
  size_t getBufferSize() const;

private:
  /**
   * @brief create Subgraph using data stored in SerializedGraphData
   * @param gd information about serializer parts of model
   * @return offset in buffer corresponding to serialized subgraph
   */
  flatbuffers::Offset<circle::SubGraph> exportSubgraph(SerializedGraphData &gd);

  /**
   * @brief root function that writes graph into internal buffer
   * @param graph
   */
  void exportGraph(loco::Graph *graph);

  /**
   * @brief root function that writes Module into internal buffer
   * @param module
   */
  void exportModule(Module *module);

private:
  flatbuffers::FlatBufferBuilder _builder;
};

} // namespace luci

#endif // __CIRCLE_EXPORTER_IMPL_H__
