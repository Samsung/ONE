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

#ifndef __NNPACKAGE_RUN_OP_OUTPUT_DUMPER_H__
#define __NNPACKAGE_RUN_OP_OUTPUT_DUMPER_H__

#include "nnfw_internal.h"

#include <string>
#include <sstream>

namespace nnpkg_run
{

class OpOutputDumper
{
public:
  static void setOutputDir(std::string output_dir)
  {
    auto ind = output_dir.length();
    if (output_dir[ind - 1] == '/')
      OpOutputDumper::_output_dir = output_dir;
    else
      OpOutputDumper::_output_dir = output_dir + "/";
  }

  static void registerDumpCallback(nnfw_session *session)
  {
    nnfw_enable_dump_op_output(session, OpOutputDumper::dumpCallback);
  }

private:
  /**
   * @brief Callback function to be called after running exec::Function
   */
  static void dumpCallback(const nnfw_output_tensor *tensor, uint32_t subgraph_ind,
                           uint32_t op_seq_ind, uint32_t op_ind, uint32_t output_ind);

  static std::string getFilename(uint32_t subgraph_ind, uint32_t op_seq_ind, uint32_t op_ind,
                                 uint32_t output_ind, uint32_t count)
  {
    std::stringstream fn;
    fn << "subg-" << subgraph_ind << "_op-" << op_ind << "_" << count << "_o-" << output_ind
       << "_opseq-" << op_seq_ind << ".h5";
    return fn.str();
  }

private:
  static std::string _output_dir;
};

} // end of namespace nnpkg_run

#endif // __NNPACKAGE_RUN_OP_OUTPUT_DUMPER_H__
