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

#ifndef __LUCI_VALUE_TEST_CIRCLEXPCONTRACT_H__
#define __LUCI_VALUE_TEST_CIRCLEXPCONTRACT_H__

#include <loco.h>
#include <luci/CircleExporter.h>
#include <luci/IR/Module.h>

#include <memory>
#include <string>

struct CircleExpContract : public luci::CircleExporter::Contract
{
public:
  CircleExpContract(luci::Module *module, const std::string &filename)
      : _module(module), _filepath(filename)
  {
    // NOTHING TO DO
  }
  virtual ~CircleExpContract() = default;

public:
  loco::Graph *graph(void) const final { return nullptr; }
  luci::Module *module(void) const final { return _module; };

public:
  bool store(const char *ptr, const size_t size) const final;

private:
  luci::Module *_module;
  const std::string _filepath;
};

#endif // __LUCI_VALUE_TEST_CIRCLEXPCONTRACT_H__
