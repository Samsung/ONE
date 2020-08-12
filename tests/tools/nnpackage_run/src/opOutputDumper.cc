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

#include "opOutputDumper.h"
#include "h5formatter.h"

#include <map>
#include <cassert>
#include <fstream>

namespace
{

class SequenceFile
{
public:
  SequenceFile(std::string filepath) : _f(filepath) {}

  void write(std::string seq)
  {
    assert(_f.is_open());
    _f << seq << std::endl;
  }

  ~SequenceFile() { _f.close(); }

private:
  std::ofstream _f;
};

} // namespace

namespace nnpkg_run
{

std::string OpOutputDumper::_output_dir = "";

void OpOutputDumper::dumpCallback(const nnfw_output_tensor *tensor, uint32_t subg_ind,
                                  uint32_t op_seq_ind, uint32_t op_ind, uint32_t output_ind)
{
  assert(tensor);
  assert(OpOutputDumper::_output_dir[OpOutputDumper::_output_dir.length() - 1] == '/');

  using SubgInd = uint32_t;
  using OpSeqInd = uint32_t;
  using OpInd = uint32_t;
  using OpRanCount = uint32_t;

  // if this op is in WHILE subgraph, this op can be ran multiple times
  // note that opIndex k in subgraph m and subgraph n can be same so we have to handle separately
  static std::map<SubgInd, std::map<OpInd, OpRanCount>> subg_map;
  std::map<OpInd, OpRanCount> *op_map;

  auto subg_found = subg_map.find(subg_ind);
  if (subg_found == subg_map.end())
  {
    subg_map[subg_ind] = std::move(std::map<OpInd, OpRanCount>());
    op_map = &subg_map[subg_ind];
  }
  else
  {
    op_map = &subg_found->second;
  }

  int count = 1;
  auto found = op_map->find(op_ind);
  if (found == op_map->end())
  {
    (*op_map)[op_ind] = count = 1; // create new count
  }
  else
  {
    count = ++(found->second); // update map
  }

  auto fn = OpOutputDumper::getFilename(subg_ind, op_seq_ind, op_ind, output_ind, count);
  TensorDumper().dumpTensor(OpOutputDumper::_output_dir + fn, tensor);

  static SequenceFile seq_file(OpOutputDumper::_output_dir + "sequence.txt");
  seq_file.write(fn);
}

} // end of namespace nnpkg_run
