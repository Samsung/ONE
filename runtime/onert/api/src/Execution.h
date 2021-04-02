/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_API_EXECUTION_H__
#define __NNFW_API_EXECUTION_H__

#include "nnfw_session.h"

namespace onert
{
namespace api
{

class Execution
{
public:
  Execution() = delete;
  Execution(nnfw_session *session);

public:
  NNFW_STATUS prepare();
  NNFW_STATUS run();
  NNFW_STATUS runAsync();
  NNFW_STATUS await();

private:
  nnfw_session *_session;
};

} // namespace api
} // namespace onert

#endif // __NNFW_API_EXECUTION_H__
