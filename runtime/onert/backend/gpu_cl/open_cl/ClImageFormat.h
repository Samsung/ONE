/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_GPU_CL_CL_IMAGE_FORMAT_H__
#define __ONERT_BACKEND_GPU_CL_CL_IMAGE_FORMAT_H__

#include "OpenclWrapper.h"
#include "DataType.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

cl_channel_order ToChannelOrder(int num_channels);

cl_channel_type ToImageChannelType(DataType data_type);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_CL_IMAGE_FORMAT_H__
