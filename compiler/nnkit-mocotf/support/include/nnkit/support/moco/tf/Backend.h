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

#ifndef __NNKIT_SUPPORT_MOCO_TF_BACKEND_H__
#define __NNKIT_SUPPORT_MOCO_TF_BACKEND_H__

#include "nnkit/Backend.h"
#include "nnkit/TensorContext.h"
#include "nnkit/support/tftestinfo/ParsedTensor.h"

#include "loco/IR/Graph.h"
#include "locomotiv/Session.h"

#include <vector>
#include <memory>

namespace nnkit
{
namespace support
{
namespace moco
{
namespace tf
{

class Backend final : public nnkit::Backend
{
  using ParsedTensors = std::vector<std::unique_ptr<nnkit::support::tftestinfo::ParsedTensor>>;

public:
  Backend(const char *pb_path, const char *info_path);

  void setInputOutputFromGraph(const std::unique_ptr<loco::Graph> &loco_graph,
                               ParsedTensors &parsed_tensors);

  void prepare(const std::function<void(nnkit::TensorContext &)> &f) override;

  void run(void) override;

  void teardown(const std::function<void(nnkit::TensorContext &)> &f) override;

private:
  std::unique_ptr<loco::Graph> _loco_graph;
  std::unique_ptr<locomotiv::Session> _sess;

  ParsedTensors _inputs;
  ParsedTensors _outputs;
};

} // namespace tf
} // namespace moco
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_MOCO_TF_BACKEND_H__
