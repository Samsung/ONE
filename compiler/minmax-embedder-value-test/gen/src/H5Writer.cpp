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

#include "H5Writer.h"
#include "Cast.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <H5Cpp.h>

namespace minmax_embedder_test
{
/*
 * ensure grp_name exists in parent
 */
H5::Group ensureGroup(H5::Group parent, const std::string &child)
{
  H5::Exception::dontPrint();
  try
  {
    return parent.openGroup(child.c_str());
  }
  catch (H5::Exception &e)
  {
    return parent.createGroup(child.c_str());
  }
}

static const char *h5_value_grpname = "value";

H5Writer::H5Writer(const ModelSpec &md_spec, const std::string &filepath)
  : _md_spec{md_spec}, _filepath{filepath}
{
}

void H5Writer::dump()
{
  // NOTE: H5Writer
  H5::H5File h5file{_filepath, H5F_ACC_CREAT | H5F_ACC_RDWR};
  auto root_grp = h5file.openGroup("/");
  ensureGroup(root_grp, h5_value_grpname);
  auto val_grp = h5file.openGroup(h5_value_grpname);
  // NOTE: Writer
  uint32_t num_run = to_u32(val_grp.getNumObjs());
  auto run_grp = val_grp.createGroup(("run_") + std::to_string(num_run));
  // Assumption: single subgraph
  auto model_grp = ensureGroup(run_grp, std::string("model_") + "0");
  hsize_t dims[] = {2};
  H5::DataSpace dspace(1, dims); // rank=1, dim(0)=2, {min, max}
  DataGen gen;
  // dump input minmax
  for (uint32_t i = 0; i < _md_spec.n_inputs; ++i)
  {
    const auto subg_idx = 0;
    auto subg_grp = ensureGroup(model_grp, std::string("subg_") + std::to_string(subg_idx).c_str());
    auto input_dset = subg_grp.createDataSet(std::string("input_") + std::to_string(i),
                                             H5::PredType::IEEE_F32BE, dspace);
    auto minmax = gen(num_run, i);
    input_dset.write(gen(num_run, i).data(), H5::PredType::NATIVE_FLOAT);
  }
  // dump op minmax
  for (uint32_t op = 0; op < _md_spec.n_ops; ++op)
  {
    const auto subg_idx = 0;
    auto subg_grp = ensureGroup(model_grp, std::string("subg_") + std::to_string(subg_idx).c_str());
    auto op_dset = subg_grp.createDataSet(std::string("op_") + std::to_string(op),
                                          H5::PredType::IEEE_F32BE, dspace);
    op_dset.write(gen(num_run, op).data(), H5::PredType::NATIVE_FLOAT);
  }
}
} // end of namespace minmax_embedder_test
