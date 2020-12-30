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
 * @file Operand.h
 * @brief    This file contains Operand
 * @ingroup  COM_AI_RUNTIME
 *
 */

#ifndef __ONERT_DUMPER_DOT_DOT_OPERAND_INFO_H__
#define __ONERT_DUMPER_DOT_DOT_OPERAND_INFO_H__

#include <vector>

#include "Node.h"
#include "ir/Operand.h"
#include "ir/Index.h"

namespace onert
{
namespace dumper
{
namespace dot
{

/**
 * @brief Class that represents an Operand
 *
 */
class Operand : public Node
{
public:
  enum class Type
  {
    UNDEFINED,
    MODEL_INPUT,
    MODEL_OUTPUT,
    INTERNAL
  };

public:
  static const std::string INPUT_SHAPE;
  static const std::string OUTPUT_SHAPE;
  static const std::string OPERAND_SHAPE;
  static const std::string BG_COLOR_SCHEME;

public:
  /**
   * @brief Construct a new Operand Node object
   *
   * @param[in] index Operand index
   * @param[in] type Operand type
   */
  Operand(const ir::OperandIndex &index, Type type);

private:
  void addBackendLabel();
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_DOT_DOT_OPERAND_INFO_H__
