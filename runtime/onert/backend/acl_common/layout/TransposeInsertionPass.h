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

#ifndef __ONERT_BACKEND_ACL_COMMON_TRANSPOSE_INSERTION_PASS_H__
#define __ONERT_BACKEND_ACL_COMMON_TRANSPOSE_INSERTION_PASS_H__

#include <ir/Graph.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

class TransposeInsertionPass
{
public:
  TransposeInsertionPass(ir::Graph &graph) : _graph{graph} {}

public:
  void run(ir::Layout backend_layout);

private:
  void insertTransposeOps(ir::Layout backend_layout);

  /**
   * @brief Insert Transpose operation that has the given 4D operand as input to permute
   *
   * @param operand_index The target operand index for the insertion
   * @param backend_layout Layout of backend
   */
  void insert4DTransposeOp(ir::OperandIndex operand_index, ir::Layout output_layout);

private:
  ir::Graph &_graph;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_TRANSPOSE_INSERTION_PASS_H__
