/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_OPS_BATCH_MATMUL_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_BATCH_MATMUL_LAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include "../ExternalContext.h"
#include "GGMLHelper.h"

#include <exec/IFunction.h>

namespace nnfw
{
namespace cker
{
class BatchMatMul;
}
} // namespace nnfw

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class BatchMatMulLayer : public ::onert::exec::IFunction
{
public:
  BatchMatMulLayer();
  ~BatchMatMulLayer();

public:
  void batchMatMulFloat32();

  void configure(const IPortableTensor *lhs, const IPortableTensor *rhs, bool adj_x, bool adj_y,
                 IPortableTensor *output, ExternalContext *ctx);

  void run() override;

private:
  void runGGML();

  const IPortableTensor *_lhs;
  const IPortableTensor *_rhs;
  IPortableTensor *_output;

  bool _adj_x;
  bool _adj_y;

  std::unique_ptr<nnfw::cker::BatchMatMul> _kernel;
  ExternalContext *_ctx;
  bool _transposed;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_BATCH_MATMUL_LAYER_H__
