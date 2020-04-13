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

#include "Convert.h"

#include <cassert>
#include <stdexcept>

namespace
{

/**
 * @note If the platform is little endian, 0x00112233 would be saved as [0x33, 0x22, 0x11, 0x00]
 *       If not, it would be saved as [0x00, 0x11, 0x22, 0x33]
 * @return Whether platform is little endian or not
 */
bool is_platform_little_endian()
{
  int32_t num = 0x00112233;
  return (*(char *)&num == 0x33);
}

} // namespace

namespace moco
{
namespace onnx
{

bool is_default_domain(const std::string domain)
{
  return (domain.compare("") == 0 || domain.compare("onnx.ai") == 0);
}

std::vector<float> get_float_data(const ::onnx::TensorProto &tensor)
{
  std::vector<float> data;

  // Exactly one of the fields is used to store the elements of the tensor
  assert(!(tensor.has_raw_data() && (tensor.float_data_size() > 0)));
  assert(tensor.has_raw_data() || (tensor.float_data_size() > 0));

  if (tensor.has_raw_data())
  {
    const std::string raw_data = tensor.raw_data();

    // If platform is big endian, we should convert data as big endian
    if (!is_platform_little_endian())
    {
      // TODO Revise implementation of this logic. This is too complex.
      const char *little_endian_bytes = raw_data.c_str();
      char *big_endian_bytes = reinterpret_cast<char *>(std::malloc(raw_data.size()));

      for (int i = 0; i < raw_data.size(); ++i)
        big_endian_bytes[i] = little_endian_bytes[i];

      const size_t element_size = sizeof(float);
      const size_t num_elements = raw_data.size() / element_size;
      for (size_t i = 0; i < num_elements; ++i)
      {
        char *start_byte = big_endian_bytes + i * element_size;
        char *end_byte = start_byte + element_size - 1;

        for (size_t count = 0; count < element_size / 2; ++count)
        {
          char temp = *start_byte;
          *start_byte = *end_byte;
          *end_byte = temp;
          ++start_byte;
          --end_byte;
        }
      }

      data.insert(data.end(), reinterpret_cast<const float *>(big_endian_bytes),
                  reinterpret_cast<const float *>(big_endian_bytes + raw_data.size()));

      std::free(big_endian_bytes);
    }
    else
    {
      const char *bytes = raw_data.c_str();
      data.insert(data.end(), reinterpret_cast<const float *>(bytes),
                  reinterpret_cast<const float *>(bytes + raw_data.size()));
    }
  }
  else
  {
    for (int i = 0; i < tensor.float_data_size(); ++i)
      data.push_back(tensor.float_data(i));
  }

  return data;
}

} // namespace onnx
} // namespace moco
