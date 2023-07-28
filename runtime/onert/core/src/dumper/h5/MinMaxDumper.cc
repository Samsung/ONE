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

#include "MinMaxDumper.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace onert
{
namespace dumper
{
namespace h5
{

static const char *h5_value_grpname = "value";

/*
 * ensure grp_name exists in parent
 */
H5::Group ensureGroup(H5::Group parent, const char *child)
{
  H5::Exception::dontPrint();
  try
  {
    return parent.openGroup(child);
  }
  catch (H5::Exception &e)
  {
    return parent.createGroup(child);
  }
}

MinMaxDumper::MinMaxDumper(const std::string &filepath) : Dumper(filepath)
{
  auto root_grp = _file.openGroup("/");
  ensureGroup(root_grp, h5_value_grpname);
}

void MinMaxDumper::dump(const exec::SMMinMaxMap &mmmap) const
{
  auto val_grp = _file.openGroup(h5_value_grpname);
  auto num_run = val_grp.getNumObjs();
  auto num_grp = val_grp.createGroup(std::to_string(num_run));
  auto model_grp = ensureGroup(num_grp, "0");
  hsize_t dims[] = {2};
  H5::DataSpace dspace(1, dims); // rank=1, dim(0)=2, {min, max}
  for (auto &&e : mmmap)
  {
    // key = {subg_idx, op_idx} = e.first
    const auto subg_idx = e.first.first.value();
    const auto op_idx = e.first.second.value();
    auto subg_grp = ensureGroup(model_grp, std::to_string(subg_idx).c_str());
    auto op_dset = subg_grp.createDataSet(std::to_string(op_idx), H5::PredType::IEEE_F32BE, dspace);
    op_dset.write(e.second.data, H5::PredType::NATIVE_FLOAT);
  }
}

} // namespace h5
} // namespace dumper
} // namespace onert
