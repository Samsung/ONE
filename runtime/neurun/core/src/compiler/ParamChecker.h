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

/**
 * @file ParamChecker.h
 * @brief This file contains ParamChecker to check\n
 *        operations' parameters are compilable at machine independent phase\n
 *        ex) Check param is constant
 */
#ifndef __NEURUN_COMPILER_PARAM_CHECKER_H__
#define __NEURUN_COMPILER_PARAM_CHECKER_H__

#include "ir/OperationVisitor.h"

namespace neurun
{
namespace ir
{
class Graph;
} // namespace ir
} // namespace neurun

namespace neurun
{
namespace compiler
{

class ParamChecker : public ir::OperationVisitor
{
public:
  /**
   * @brief Construct a new Param Checker object (deleted)
   */
  ParamChecker(void) = delete;
  /**
   * @brief Construct a new Param Checker object
   * @param[in] model Graph model to check
   */
  ParamChecker(std::shared_ptr<ir::Graph> model) : _model{model} {}

public:
  /**
   * @brief Run parameter analysis
   */
  void operator()();
  /**
   * @brief   Return analysis result if model have non-const parameter
   * @return  @c true if there is non-const parameter, otherwise @c false
   */
  bool haveNoneConstParam(void) { return _nonConstParam; }

private:
  const std::shared_ptr<ir::Graph> _model;
  bool _nonConstParam{false};
};

} // namespace compiler
} // namespace neurun

#endif // __NEURUN_COMPILER_OPERATION_VALIDATOR_H__
