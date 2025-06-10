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
#ifndef __ONERT_EXEC_NOP_FUNCTION_H_
#define __ONERT_EXEC_NOP_FUNCTION_H_

#include "IFunction.h"

namespace onert::exec
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
};

} // namespace onert::exec

#endif // __ONERT_EXEC_NOP_FUNCTION_H_
