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

#include "Reader.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace minmax
{
namespace h5
{
static const char *h5_value_grpname = "value";

bool exists(hid_t id, const char *path) { return H5Lexists(id, path, H5P_DEFAULT) > 0; }

Reader::Reader(const std::string &filepath) : _file(filepath, H5F_ACC_RDONLY)
{
  _val_grp = _file.openGroup(h5_value_grpname);
}

// TODO: Handle multiple output
MinMaxVectors Reader::read(int model_idx, int subg_idx, int op_idx) const
{
  MinMaxVectors mmv;
  float minmax[2];
  auto num_run = _val_grp.getNumObjs();
  for (uint32_t r = 0; r < num_run; ++r)
  {
    // check whether minmax exists
    char path[128]; // enough to print 4 x { '-' (1) int64 (18) + '/' (1) } + null
    snprintf(path, 128, "/value/%d/%d/%d/%d", r, model_idx, subg_idx, op_idx);
    if (!exists(_file.getId(), path))
      continue;
    auto run_grp = _val_grp.openGroup(std::to_string(r));
    auto model_grp = run_grp.openGroup(std::to_string(model_idx));
    auto subg_grp = model_grp.openGroup(std::to_string(subg_idx));
    auto op_dset = subg_grp.openDataSet(std::to_string(op_idx));
    H5::DataType dtype = op_dset.getDataType();
    if (not(dtype == H5::PredType::IEEE_F32BE || dtype == H5::PredType::IEEE_F32LE))
      throw std::runtime_error{"dtype of min, max in h5 is not float."};
    op_dset.read(minmax, H5::PredType::NATIVE_FLOAT);
    mmv.min_vector.emplace_back(minmax[0]);
    mmv.max_vector.emplace_back(minmax[1]);
  }
  return mmv;
}

} // namespace h5
} // namespace minmax
