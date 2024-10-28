/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "record-hessian/HessianComputer.h"

#include <luci/IR/CircleQuantParam.h>

namespace record_hessian
{

/**
 * @brief unfold the vector with NHWC shape, inherently acting in an in-place manner.
 * @note (N, H, W, C) -> (N, L, H*W*C)
 */
void HessianComputer::unfold(std::vector<float> &buf, uint32_t input_n, uint32_t input_h,
                             uint32_t input_w, uint32_t input_c, uint32_t stride_h,
                             uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
                             uint32_t kernel_oc, uint32_t kernel_h, uint32_t kernel_w,
                             uint32_t kernel_ic)
{
  if (input_c != kernel_ic) {
    throw std::runtime_error("Input channels do not match kernel channels.");
  }
  int out_height = (input_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  int out_width = (input_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
  int patch_size = kernel_h * kernel_w * kernel_ic;
  std::vector<float> unfolded_buf(input_n * out_height * out_width * patch_size, 0.0f);

  int index = 0;
  int in_y, in_x;
  for (int n = 0; n < input_n; ++n)
  {
    for (int y = 0; y < out_height; ++y)
    {
      for (int x = 0; x < out_width; ++x)
      {
        for (int in_c = 0; in_c < input_c; ++in_c)
        {
          for (int ky = 0; ky < kernel_h; ++ky)
          {
            for (int kx = 0; kx < kernel_w; ++kx)
            {
              in_y = y * stride_h + ky * dilation_h;
              in_x = x * stride_w + kx * dilation_w;
              if (in_y < input_h && in_x < input_w)
              {
                unfolded_buf[index] = buf[((n * input_h + in_y) * input_w + in_x) * input_c + in_c];
              }
              index++;
            }
          }
        }
      }
    }
  }

  buf.swap(unfolded_buf);
}

void HessianComputer::recordHessianForFullyConnected(const luci::CircleNode *node, 
                                                     const luci_interpreter::Tensor *input_tensor)
{
  uint32_t size_in_ch;
  uint32_t length;

  const auto data = input_tensor->data<float>();
  const auto num_elements = input_tensor->shape().num_elements();
  std::vector<float> buf(data, data + num_elements);

  // get the size of input channel from weights
  if (input_tensor->shape().num_dims() == 3)
  { // input_tensor [batch, length, channel]
    size_in_ch = input_tensor->shape().dim(2);
  }
  else if (input_tensor->shape().num_dims() == 2)
  {
    size_in_ch = input_tensor->shape().dim(1); // input_tensor [length, channel]
  }
  else
  {
    throw std::runtime_error("Unsupported node rank");
  }
  length = num_elements / size_in_ch;

  std::vector<float> hessian(size_in_ch * size_in_ch, 0);

  for (int i = 0; i < size_in_ch; ++i)
  {
    for (int j = 0; j < size_in_ch; ++j)
    {
      float sum = 0;
      for (int k = 0; k < length; ++k)
      {
        sum += buf[i + k * size_in_ch] * buf[j + k * size_in_ch];
      }
      hessian[i * size_in_ch + j] = 2 * sum;
    }
  }

  HessianVector &vector = _hessian_map[node];
  vector.update(hessian);
}

void HessianComputer::recordHessianForConv2D(const luci::CircleNode *node,
                                             const luci_interpreter::Tensor *input_tensor)
{
  const auto node_filter = loco::must_cast<luci::CircleConst *>(
      loco::must_cast<const luci::CircleConv2D *>(node)->filter());
  const auto node_bias = loco::must_cast<luci::CircleConst *>(
      loco::must_cast<const luci::CircleConv2D *>(node)->bias());
  uint32_t size_in_ch =
      node_filter->size<loco::DataType::FLOAT32>() / node_bias->size<loco::DataType::FLOAT32>();

  uint32_t input_n = input_tensor->shape().dim(0);
  uint32_t input_h = input_tensor->shape().dim(1);
  uint32_t input_w = input_tensor->shape().dim(2);
  uint32_t input_c = input_tensor->shape().dim(3);

  uint32_t stride_h = ((luci::CircleConv2D *)node)->stride()->h();
  uint32_t stride_w = ((luci::CircleConv2D *)node)->stride()->w();
  uint32_t dilation_h = ((luci::CircleConv2D *)node)->dilation()->h();
  uint32_t dilation_w = ((luci::CircleConv2D *)node)->dilation()->w();

  uint32_t kernel_oc = node_filter->dim(0).value();
  uint32_t kernel_h = node_filter->dim(1).value();
  uint32_t kernel_w = node_filter->dim(2).value();
  uint32_t kernel_ic = node_filter->dim(3).value();

  const auto data = input_tensor->data<float>();
  const auto num_elements = input_tensor->shape().num_elements();
  std::vector<float> buf(data, data + num_elements);

  unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h, dilation_w,
          kernel_oc, kernel_h, kernel_w, kernel_ic);
  uint32_t length = buf.size() / size_in_ch;

  std::vector<float> hessian(size_in_ch * size_in_ch, 0);
  for (int i = 0; i < size_in_ch; ++i)
  {
    for (int j = 0; j < size_in_ch; ++j)
    {
      float sum = 0;
      for (int k = 0; k < length; ++k)
      {
        sum += buf[i + k * size_in_ch] * buf[j + k * size_in_ch];
      }
      hessian[i * size_in_ch + j] = 2 * sum;
    }
  }

  HessianVector &vector = _hessian_map[node];
  vector.update(hessian);
}

void HessianComputer::recordHessian(const luci::CircleNode *node,
                                    const luci_interpreter::Tensor *input_tensor)
{
  if (node == nullptr || input_tensor == nullptr)
  {
    throw std::invalid_argument("node or input_tensor is null.");
  }

  if (input_tensor->element_type() != loco::DataType::FLOAT32)
  {
    throw std::runtime_error("Unsupported dtype: only FLOAT32 is supported.");
  }

  if (node->opcode() == luci::CircleOpcode::FULLY_CONNECTED)
  {
    recordHessianForFullyConnected(node, input_tensor);
  }
  else if (node->opcode() == luci::CircleOpcode::CONV_2D)
  {
    recordHessianForConv2D(node, input_tensor);
  }
  else{
    throw std::runtime_error(node->name() + " is unsupported op for record hessian.");
  }
}

} // namespace record_hessian
