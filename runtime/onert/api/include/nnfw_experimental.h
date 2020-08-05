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

#ifndef __NNFW_EXPERIMENTAL_H__
#define __NNFW_EXPERIMENTAL_H__

#include "nnfw.h"

// Used for custom kernel development

/*
 * operand type, used only for custom operations
 */
typedef struct
{
  nnfw_tensorinfo type;
  void *allocation;
} nnfw_operand;

/*
 * Used as input to custom operation eval function
 */
typedef struct
{
  size_t ninputs;
  nnfw_operand *inputs;

  size_t noutputs;
  nnfw_operand *outputs;
} nnfw_custom_kernel_params;

/*
 * Custom kernel evaluation function
 *
 * param[in] params custom operation parameters
 * param[in] userdata pointer to user-specified buffer( kernel instance specific )
 */
typedef void (*nnfw_custom_eval)(nnfw_custom_kernel_params *params, char *userdata,
                                 size_t userdata_size);

/*
 * custom operation registration info
 */
typedef struct
{
  nnfw_custom_eval eval_function;
} custom_kernel_registration_info;

NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info);

#endif // __NNFW_EXPERIMENTAL_H__
