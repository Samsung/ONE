/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "randomgen.h"
#include "nnfw_util.h"
#include "misc/RandomGenerator.h"

#include <iostream>

namespace onert_train
{

// template <class T> void randomData(nnfw::misc::RandomGenerator &randgen, void *data, uint64_t
// size)
// {
//   for (uint64_t i = 0; i < size; i++)
//     reinterpret_cast<T *>(data)[i] = randgen.generate<T>();
// }

// Generator RandomGenerator::generate(std::vector<nnfw_tensorinfo> &input_infos,
// std::vector<nnfw_tensorinfo> &output_infos, uint32_t size)
// {
//   // generate random data
//   const int seed = 1;
//   nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

//   auto input_size = std::accumulate(input_infos.begin(), input_infos.end(), 0, bufsize_for);
//   auto output_size = std::accumulate(output_infos.begin(), output_infos.end(), 0, bufsize_for);
//   return [randgen, input_size, output_size, size, this](uint32_t idx, std::vector<Allocation>
//   &inputs,
//                          std::vector<Allocation> &expecteds) {
//       for (uint32_t d = 0; d < data_length; ++d)
//       {
//         const auto &input = inputs[d*num_inputs+i];
//         input.alloc(input_size_in_bytes);
//         switch (ti.dtype)
//         {
//           case NNFW_TYPE_TENSOR_FLOAT32:
//             randomData<float>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
//             randomData<uint8_t>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_BOOL:
//             randomData<bool>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_UINT8:
//             randomData<uint8_t>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_INT32:
//             randomData<int32_t>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_INT64:
//             randomData<int64_t>(randgen, input.data(), input_num_elems);
//             break;
//           case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
//             randomData<int16_t>(randgen, input.data(), input_num_elems);
//             break;
//           default:
//             std::cerr << "Not supported input type" << std::endl;
//             std::exit(-1);
//         }
//       }
//     }

//   }
// };

} // end of namespace onert_train
