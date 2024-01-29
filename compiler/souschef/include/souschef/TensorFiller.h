/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __SOUSCHEF_TENSOR_FILLER_H__
#define __SOUSCHEF_TENSOR_FILLER_H__

#include <map>
#include <vector>

namespace souschef
{

class TensorFiller
{
public:
  virtual ~TensorFiller() = default;

  /**
   * @brief This will record the tensor by index, if it needs filler option,
   *        such as kernel, bias.
   */
  void set_tensor_filler(uint32_t tensor_index) { _tensor_filler[tensor_index] = true; }

  /**
   * @brief This will store int32 filler values such as reshape information for the tensor
   */
  void set_tensor_filler(uint32_t tensor_index, std::vector<int32_t> &expvalues)
  {
    _tensor_filler_vint32[tensor_index] = expvalues;
  }

  void set_tensor_filler(uint32_t tensor_index, std::vector<float> &expvalues)
  {
    _tensor_filler_vfloat[tensor_index] = expvalues;
  }

  /**
   * @brief This will return true if the tensor by index, needs a filler option.
   */
  bool get_tensor_filler(uint32_t tensor_index)
  {
    auto it = _tensor_filler.find(tensor_index);
    if (it != _tensor_filler.end())
    {
      return it->second;
    }
    return false;
  }

  /**
   * @brief This will return true if the tensor by index, needs a int array filler option.
   */
  bool get_tensor_filler(uint32_t tensor_index, std::vector<int32_t> &expvalues)
  {
    auto it = _tensor_filler_vint32.find(tensor_index);
    if (it != _tensor_filler_vint32.end())
    {
      expvalues = it->second;
      return true;
    }
    return false;
  }

  bool get_tensor_filler(uint32_t tensor_index, std::vector<float> &expvalues)
  {
    auto it = _tensor_filler_vfloat.find(tensor_index);
    if (it != _tensor_filler_vfloat.end())
    {
      expvalues = it->second;
      return true;
    }
    return false;
  }

  void clear_tensor_filler() { _tensor_filler.clear(); }

  void clear_tensor_filler_vint32() { _tensor_filler_vint32.clear(); }

  void clear_tensor_filler_vfloat() { _tensor_filler_vfloat.clear(); }

private:
  std::map<uint32_t, bool> _tensor_filler{};
  std::map<uint32_t, std::vector<int32_t>> _tensor_filler_vint32{};
  std::map<uint32_t, std::vector<float>> _tensor_filler_vfloat{};
};

} // namespace souschef

#endif // __SOUSCHEF_TENSOR_FILLER_H__
