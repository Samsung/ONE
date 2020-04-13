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

#ifndef __NNKIT_SUPPORT_TF_BACKEND_H__
#define __NNKIT_SUPPORT_TF_BACKEND_H__

#include "nnkit/support/tf/TensorDataMap.h"
#include "nnkit/support/tf/TensorContext.h"
#include "nnkit/support/tf/Runner.h"
#include "nnkit/support/tftestinfo/ParsedTensor.h"

#include <nnkit/Backend.h>

#include <memory>
#include <vector>

namespace nnkit
{
namespace support
{
namespace tf
{

using nnkit::support::tftestinfo::ParsedTensor;

class Backend final : public nnkit::Backend
{
public:
  Backend() = delete;
  Backend(const Backend &) = delete;
  Backend(Backend &&) = delete;

  Backend(const char *pb_path, const char *info_path);

  void prepare(const std::function<void(nnkit::TensorContext &)> &f) override;

  void run(void) override;

  void teardown(const std::function<void(nnkit::TensorContext &)> &f) override;

private:
  std::vector<std::unique_ptr<ParsedTensor>> _inputs;
  std::vector<std::unique_ptr<ParsedTensor>> _outputs;

  TensorDataMap _data_map;

  Runner _tf_runner;
};

} // namespace tf
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TF_BACKEND_H__
