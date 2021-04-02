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

#ifndef __NNFW_API_TENSOR_MANAGER_H__
#define __NNFW_API_TENSOR_MANAGER_H__

#include "nnfw_session.h"

namespace onert
{
namespace api
{

class TensorManager
{
public:
  TensorManager() = delete;
  TensorManager(nnfw_session *session);

public:
  NNFW_STATUS setInput(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS setOutput(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS inputSize(uint32_t *number);
  NNFW_STATUS outputSize(uint32_t *number);

  NNFW_STATUS setInputLayout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS setOutputLayout(uint32_t index, NNFW_LAYOUT layout);

  NNFW_STATUS applyTensorinfo(uint32_t index, nnfw_tensorinfo ti); // Will be deprecated
  NNFW_STATUS setInputTensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS inputTensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS outputTensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  NNFW_STATUS inputTensorindex(const char *tensorname, uint32_t *index);
  NNFW_STATUS outputTensorindex(const char *tensorname, uint32_t *index);

private:
  nnfw_session *_session;
};

} // namespace api
} // namespace onert

#endif // __NNFW_API_TENSOR_MANAGER_H__
