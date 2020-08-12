/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_I_FUNCTION_H__
#define __ONERT_EXEC_I_FUNCTION_H__

#include "backend/ITensor.h"
#include "util/Utils.h"
#include "ir/Index.h"

#include <stdexcept>
#include <vector>

namespace onert
{
namespace exec
{

class IFunction
{
public:
  virtual ~IFunction() = default;
  virtual void run() = 0;
  virtual void prepare() {}

  /**
   * @brief Get the number of outputs. Currently this is used by observer after running Function.
   *        By default it returns 1. If number of outputs is more than 1, override this method.
   */
  virtual uint32_t getOutputSize() const { return 1; }

  /**
   * @brief Get the Outputs object. Currently this is used by observer after running Function.
   */
  virtual const backend::ITensor *getOutput(int output_ind = 0) const
  {
    UNUSED_RELEASE(output_ind);
    throw std::runtime_error("Not Supported");
  }

  struct UniqueID
  {
    ir::SubgraphIndex subg_ind;
    ir::OpSequenceIndex op_seq_ind;
    ir::OperationIndex op_ind;

    void set(ir::SubgraphIndex subg, ir::OpSequenceIndex op_seq, ir::OperationIndex op)
    {
      subg_ind = subg;
      op_seq_ind = op_seq;
      op_ind = op;
    }

    bool isSame(ir::SubgraphIndex subg, ir::OpSequenceIndex op_seq, ir::OperationIndex op)
    {
      assert(subg.valid() && op_seq.valid() && op.valid());
      assert(subg_ind.valid() && op_seq_ind.valid() && op_ind.valid());
      return subg == subg_ind && op_seq == op_seq_ind && op == op_ind;
    }
  };
  UniqueID id;

  const std::string &name() const { return _name; }
  void name(const std::string &op_name) { _name = op_name; }
private:
  std::string _name;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_FUNCTION_H__
