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

#ifndef __NNFW_API_CONFIG_H__
#define __NNFW_API_CONFIG_H__

#include "nnfw_api_internal.h"

namespace onert
{
namespace api
{

class Config
{
public:
  Config() = delete;
  Config(nnfw_session *session);

public:
  NNFW_STATUS setAvailableBackends(const char *backends);
  NNFW_STATUS setOpBackend(const char *op, const char *backend);

  NNFW_STATUS setConfig(const char *key, const char *value);
  NNFW_STATUS getConfig(const char *key, char *value, size_t value_size);

  NNFW_STATUS registerCustomOperation(const std::string &id, nnfw_custom_eval eval_func);

private:
  nnfw_session *_session;
};

} // namespace api
} // namespace onert

#endif // __NNFW_API_CONFIG_H__
