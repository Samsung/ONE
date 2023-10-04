/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MINMAX_EMBEDDER_H5_READER_H__
#define __MINMAX_EMBEDDER_H5_READER_H__

#include <H5Cpp.h>
#include <string>
#include <utility>
#include <vector>

namespace minmax
{
namespace h5
{
// The hierachy of single model minmax h5 file
//
// GROUP /
//   GROUP value
//     └── GROUP run_{idx}
//           └── GROUP model_{idx}
//                 └── GROUP subg_{idx}
//                       ├── DATASET op_{idx}
//                       │      DATATYPE Float32
//                       │      DATASPACE (2)
//                       │      DATA { min, max }
//                       └── DATASET input_{idx}
//                              DATATYPE Float32
//                              DATASPACE (2)
//                              DATA { min, max }
//   GROUP name   (optional, for debug)
//     └── GROUP model_{idx}
//           └── GROUP subg_{idx}
//                       ├── ATTRIBUTE op_{idx}
//                       │      DATATYPE String
//                       │      DATA { "op/name"}

//                       └── ATTRIBUTE input_{idx}
//                              DATATYPE String
//                              DATA { "input/name"}
struct MinMaxVectors
{
  std::vector<float> min_vector;
  std::vector<float> max_vector;
};

class Reader
{
public:
  Reader(const std::string &filepath);
  /**
   * @brief Returns minmax recording for op {model_idx, subg_idx, op_idx}
   *
   * @return MinMaxVectors
   */
  MinMaxVectors read(int model_idx, int subg_idx, int op_idx) const;
  /**
   * @brief Returns minmax recording for input {model_idx, subg_idx, input_idx}
   *
   * @return MinMaxVectors
   */
  MinMaxVectors read_input(int model_idx, int subg_idx, int input_idx) const;

private:
  H5::H5File _file;
  H5::Group _val_grp;
};

} // namespace h5
} // namespace minmax

#endif // __MINMAX_EMBEDDER_H5_READER_H__
