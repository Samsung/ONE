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

#ifndef __ONERT_BACKEND_CPU_OPS_CONVOLUTIONLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_CONVOLUTIONLAYER_H__

#include "../Tensor.h"
#include "OperationUtils.h"

#include <exec/IFunction.h>
#include <functional>
#include <memory>

namespace nnfw
{
namespace cker
{
class Conv;
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

class ConvolutionLayer : public ::onert::exec::IFunction
{
public:
  ConvolutionLayer();
  ~ConvolutionLayer();

public:
  void convFloat32();

  void convQuant8();

  void configure(const Tensor *input, const Tensor *kernel, const Tensor *bias,
                 const ir::Padding padding, const uint32_t paddingLeft, const uint32_t paddingRight,
                 const uint32_t paddingTop, const uint32_t paddingBottom,
                 const uint32_t strideWidth, const uint32_t strideHeight,
                 const ir::Activation activation, Tensor *output);
  void configure(const Tensor *input, const Tensor *kernel, const Tensor *bias,
                 const ir::Padding padding, const ir::Activation activation,
                 const uint32_t strideWidth, const uint32_t strideHeight,
                 const ir::Layout current_op_seq_layout, Tensor *output);
  void run();

private:
  const Tensor *_input;
  const Tensor *_kernel;
  const Tensor *_bias;
  Tensor *_output;

  ir::Layout _current_op_seq_layout;
  ir::Padding _padding;

  uint32_t _paddingLeft;
  uint32_t _paddingTop;
  uint32_t _paddingRight;
  uint32_t _paddingBottom;

  uint32_t _strideWidth;
  uint32_t _strideHeight;

  ir::Activation _activation;

  std::unique_ptr<nnfw::cker::Conv> _conv_kernel;

  bool _prepare;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_CONVOLUTIONLAYER_H__
