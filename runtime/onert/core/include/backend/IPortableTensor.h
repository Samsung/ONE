/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_I_PORTABLE_TENSOR_H__
#define __ONERT_BACKEND_I_PORTABLE_TENSOR_H__

#include "backend/ITensor.h"

namespace onert
{
namespace backend
{

/**
 * @brief A tensor class that is portable for other backends
 *
 * Backends that use derivatives of this interface can reuse each other's tensors without copying.
 * Here's criterion to be a portable tensor:
 *   - it must not have any paddings
 *   - No special operations on @c access method
 *     - e.g. CL memory must map/unmap to use it from CPU, the memory so it cannot be portable
 */
class IPortableTensor : public ITensor
{
public:
  virtual ~IPortableTensor() = default;
  virtual bool is_sparse() const { return false; }
  virtual const uint16_t *w1_segments() const { return nullptr; }
  virtual const uint16_t *w1_indices() const { return nullptr; }

public:
  bool has_padding() const final { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final { fn(*this); }
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_I_PORTABLE_TENSOR_H__
