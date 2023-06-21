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

/**
 * @file Operation.h
 * @brief    This file contains Operation
 * @ingroup  COM_AI_RUNTIME
 *
 */

#ifndef __ONERT_DUMPER_DOT_DOT_NODE_INFO_H__
#define __ONERT_DUMPER_DOT_DOT_NODE_INFO_H__

#include "Node.h"
#include "ir/IOperation.h"
#include "ir/Index.h"

namespace onert
{
namespace dumper
{
namespace dot
{

/**
 * @brief Class that represents an Operation
 *
 */
class Operation : public Node
{
public:
  static const std::string OPERATION_SHAPE;
  static const std::string BG_COLOR_SCHEME;

public:
  /**
   * @brief Construct a new Operation Node object
   *
   * @param[in] index operation index
   * @param[in] node operation object
   */
  Operation(const ir::OperationIndex &index, const ir::IOperation &node);
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_DOT_NODE_INFO_H__
