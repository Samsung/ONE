/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MOCO_FRONTEND_ONNX_ONNXUTIL_H__
#define __MOCO_FRONTEND_ONNX_ONNXUTIL_H__

#include <onnx/onnx.pb.h>

#include <string>

namespace moco
{
namespace onnx
{

/**
 * @brief If domain is empty string or onnx.ai, it is default domain
 * @param [in] domain The name of domain
 * @return Whether it is default domain or not
 */
bool is_default_domain(const std::string domain);

/**
 * @brief Get float tensor data
 * @param [in] tensor Tensor to get float data
 * @return Float vector which stores float tensor data
 */
std::vector<float> get_float_data(const ::onnx::TensorProto &tensor);

} // namespace onnx
} // namespace moco

#endif // __MOCO_FRONTEND_ONNX_ONNXUTIL_H__
