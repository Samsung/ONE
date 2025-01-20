/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_ARRAY_INDEX_H__
#define __LUCI_ARRAY_INDEX_H__

#include <cstdint>

namespace luci
{

/**
 * @brief Index class for 4D tensor (NHWC) to calculate linear index from multi-dimensional indices.
 */
class Array4DIndex
{
public:
  Array4DIndex(uint32_t N, uint32_t H, uint32_t W, uint32_t D);
  /**
   * @brief Calculate linear index from multi-dimensional indices.
   */
  uint32_t operator()(uint32_t n, uint32_t h, uint32_t w, uint32_t d) const;
  /**
   * @brief Get total number of elements in the tensor.
   */
  uint32_t size(void) const;
  /**
   * @brief Get stride of the given axis.
   */
  uint32_t stride(uint32_t axis) const;

protected:
  uint32_t _dim[4];
  uint32_t _strides[4];
};

} // namespace luci

#endif // __LUCI_ARRAY_INDEX_H__
