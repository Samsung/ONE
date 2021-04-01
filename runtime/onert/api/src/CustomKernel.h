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

#ifndef __ONERT_API_CUSTOM_KERNEL_H__
#define __ONERT_API_CUSTOM_KERNEL_H__

#include "nnfw_experimental.h"

#include "backend/CustomKernelBuilder.h"
#include "exec/IFunction.h"

#include <vector>

namespace onert
{
namespace api
{

class CustomKernel : public ::onert::exec::IFunction
{
public:
  explicit CustomKernel(nnfw_custom_eval evalFunction);

  backend::custom::CustomKernelConfigParams _in_params;

  char *_userdata;
  size_t _userdata_size;

  nnfw_custom_eval _evalFunction;
  // nnfw_custom_type_infer _type_infer_function; //Unused for now

  /**
   * Fills _params field used later by user specified eval function
   * @param inParams custom kernel parameters
   */
  virtual void configure(backend::custom::CustomKernelConfigParams &&inParams);

  void run() override;
};

} // namespace api
} // namespace onert

#endif // __ONERT_API_CUSTOM_KERNEL_H__
