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
 * @file  NopFunction.h
 * @brief This file defines NopFunction
 */
#ifndef __NEURUN_EXEC_NOP_FUNCTION_H_
#define __NEURUN_EXEC_NOP_FUNCTION_H_

#include "IFunction.h"

namespace neurun
{
namespace exec
{

/**
 * @brief A derivative of IFunction tha does nothing
 *
 */
class NopFunction : public IFunction
{
public:
  NopFunction() = default;
  void run() override
  {
    // DO NOTHING
  }
  void runSync() override
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }
};

} // namespace exec
} // namespace neurun

#endif // __NEURUN_EXEC_NOP_FUNCTION_H_
