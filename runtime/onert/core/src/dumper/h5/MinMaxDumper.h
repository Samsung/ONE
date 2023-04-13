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

#ifndef __ONERT_DUMPER_H5_MINMAX_DUMPER_H__
#define __ONERT_DUMPER_H5_MINMAX_DUMPER_H__

#include "exec/MinMaxMap.h"
#include "Dumper.h"

#include <H5Cpp.h>
#include <string>

namespace onert
{
namespace dumper
{
namespace h5
{

// The hierachy of single model minmax h5 file
//
// GROUP /
//   GROUP value
//     └── GROUP run_idx
//           └── GROUP model_idx
//                 └── GROUP subg_idx
//                       └── DATASET op_idx
//                              DATATYPE Float32
//                              DATASPACE (2)
//                              DATA { min, max }
//   GROUP name   (optional, for debug)
//     └── GROUP model_idx
//           └── GROUP subg_idx
//                 └── ATTRIBUTE op_idx
//                        DATATYPE String
//                        DATA { "model/your/op/name"}
//
class MinMaxDumper : private Dumper
{
public:
  MinMaxDumper(const std::string &filepath);
  /**
   * @brief Dump minmax map
   *
   * @param[in] map  single model minmax map
   */
  void dump(const exec::SMMinMaxMap &map) const;

private:
  H5::Group _val_grp;
};

} // namespace h5
} // namespace dumper
} // namespace onert

#endif // __ONERT_DUMPER_H5_MINMAX_DUMPER_H__
