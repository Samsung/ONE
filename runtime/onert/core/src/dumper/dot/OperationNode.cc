/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <sstream>

#include "OperationNode.h"
#include "ir/Graph.h"
#include "backend/IConfig.h"
#include "backend/Backend.h"

namespace onert
{
namespace dumper
{
namespace dot
{

const std::string Operation::OPERATION_SHAPE = "rect";
const std::string Operation::BG_COLOR_SCHEME = "pastel18";

Operation::Operation(const ir::OperationIndex &index, const ir::IOperation &node)
  : Node{"operation" + std::to_string(index.value())}
{
  setAttribute("label", std::to_string(index.value()) + " : " + node.name());
  setAttribute("shape", OPERATION_SHAPE);
  setAttribute("colorscheme", BG_COLOR_SCHEME);
  setAttribute("fillcolor", DEFAULT_FILLCOLOR);
}

} // namespace dot
} // namespace dumper
} // namespace onert
