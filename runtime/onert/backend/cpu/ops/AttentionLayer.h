/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_OPS_ATTENTION_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_ATTENTION_LAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert::backend::cpu::ops
{

class AttentionLayer : public ::onert::exec::IFunction
{
public:
  AttentionLayer();
  ~AttentionLayer();

public:
  void configure(const IPortableTensor *input, const IPortableTensor *wq, const IPortableTensor *wk,
                 const IPortableTensor *wv, const IPortableTensor *wo, const IPortableTensor *cos,
                 const IPortableTensor *sin, const IPortableTensor *mask, IPortableTensor *k_cache,
                 IPortableTensor *v_cache, const IPortableTensor *pos, IPortableTensor *output);

  void run() override;

private:
  void attentionFloat32();

private:
  const IPortableTensor *_input;
  const IPortableTensor *_wq;
  const IPortableTensor *_wk;
  const IPortableTensor *_wv;
  const IPortableTensor *_wo;
  const IPortableTensor *_cos;
  const IPortableTensor *_sin;
  const IPortableTensor *_mask;
  IPortableTensor *_k_cache;
  IPortableTensor *_v_cache;
  const IPortableTensor *_cache_pos;
  IPortableTensor *_output;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_ATTENTION_LAYER_H__
