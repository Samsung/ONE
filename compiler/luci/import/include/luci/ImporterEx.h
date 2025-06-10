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

#ifndef __LUCI_IMPORTER_EX_H__
#define __LUCI_IMPORTER_EX_H__

#include "luci/IR/Module.h"

// NOTE we should include "luci/Import/GraphBuilderRegistry.h" but
// tizen_gbs build fails if included
// TODO enable include and remove forward declaration
// #include "luci/Import/GraphBuilderRegistry.h"
struct GraphBuilderSource;

#include <functional>
#include <memory>
#include <string>

namespace luci
{

class ImporterEx final
{
public:
  ImporterEx();
  ImporterEx(const std::function<void(const std::exception &)> &error_handler);

public:
  // TODO remove this after embedded-import-value-test has moved to onert-micro
  explicit ImporterEx(const GraphBuilderSource *source);

public:
  std::unique_ptr<Module> importVerifyModule(const std::string &input_path) const;

  // NOTE importModule is for embedded-import-value-test
  // embedded-import-value-test uses constant data from file(actually ROM)
  // so unloading file will break the precondition
  // TODO remove this after embedded-import-value-test has moved to onert-micro
  std::unique_ptr<Module> importModule(const std::vector<char> &model_data) const;

private:
  const GraphBuilderSource *_source = nullptr;
  std::function<void(const std::exception &)> _error_handler;
};

} // namespace luci

#endif // __LUCI_IMPORTER_EX_H__
