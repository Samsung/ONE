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

#ifndef __LUCI_CIRCLEEXPORTER_H__
#define __LUCI_CIRCLEEXPORTER_H__

#include <luci/IR/Module.h>

#include <loco.h>

#include <memory>

namespace luci
{

class CircleExporter
{
public:
  // This contract class describes the interaction between a exporter and its client.
  struct Contract
  {
  public:
    virtual ~Contract() = default;

  public: // Client -> Exporter
    // Input Graph (to be exported)
    // Exporter expects a loco graph that consists of Circle nodes
    virtual loco::Graph *graph(void) const = 0;

    // Input Module (to be exported)
    // Exporter expects a luci module that consists of loco graphs
    // TODO make this pure virtual
    virtual luci::Module *module(void) const;

  public: // Exporter -> Client
    // Exporter calls store for export data
    // Notice: Please DO NOT STORE ptr and size when implementing this in Client
    virtual bool store(const char *ptr, const size_t size) const = 0;
  };

public:
  explicit CircleExporter();

public:
  // invoke(...) returns false on failure.
  bool invoke(Contract *) const;
};

} // namespace luci

#endif // __LUCI_CIRCLEEXPORTER_H__
