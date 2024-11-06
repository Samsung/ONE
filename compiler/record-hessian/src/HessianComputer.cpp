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
 * @note (N, H, W, C) -> (N, L, K_h * K_w * C).
 * See details(https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html).
 */
void unfold(std::vector<float> &buf, uint32_t input_n, uint32_t input_h, uint32_t input_w,
            uint32_t input_c, uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h,
            uint32_t dilation_w, uint32_t kernel_h, uint32_t kernel_w, uint32_t kernel_ic)
{
  assert(input_n > 0 && input_h > 0 && input_w > 0 && input_c > 0);
  assert(stride_h > 0 && stride_w > 0);
  assert(kernel_h > 0 && kernel_w > 0 && kernel_ic > 0);

  if (input_c != kernel_ic)
    throw std::runtime_error("RecordHessian: Input channels do not match kernel channels.");
  uint32_t out_height = (input_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  uint32_t out_width = (input_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
  uint32_t patch_size = kernel_h * kernel_w * kernel_ic;
  std::vector<float> unfolded_buf(input_n * out_height * out_width * patch_size, 0.0f);

  uint32_t index = 0;
  uint32_t in_y, in_x;
  for (uint32_t n = 0; n < input_n; ++n)
  {
    for (uint32_t y = 0; y < out_height; ++y)
    {
      for (uint32_t x = 0; x < out_width; ++x)
      {
        for (uint32_t in_c = 0; in_c < input_c; ++in_c)
        {
          for (uint32_t ky = 0; ky < kernel_h; ++ky)
          {
            for (uint32_t kx = 0; kx < kernel_w; ++kx)
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

void HessianComputer::recordHessianForFullyConnected(const luci::CircleNode *node)
{
  assert(_input_tensor->shape().num_dims() < 4);
  assert(_input_tensor->element_type() == luci_interpreter::DataType::FLOAT32);

  uint32_t size_in_ch;
  uint32_t length;

  const auto data = _input_tensor->data<float>();
  const auto num_elements = _input_tensor->shape().num_elements();
  std::vector<float> buf(data, data + num_elements);

  if (_input_tensor->shape().num_dims() == 3)
  {
    size_in_ch = _input_tensor->shape().dim(2); // input_tensor [batch, length, channel]
  }
  else if (_input_tensor->shape().num_dims() == 2)
  {
    size_in_ch = _input_tensor->shape().dim(1); // input_tensor [length, channel]
  }
  else
  {
    throw std::runtime_error("RecordHessian: Unsupported node rank");
  }
  assert(size_in_ch != 0);
  length = num_elements / size_in_ch;

  std::vector<float> hessian(size_in_ch * size_in_ch, 0);

  for (uint32_t i = 0; i < size_in_ch; ++i)
  {
    for (uint32_t j = 0; j < size_in_ch; ++j)
    {
      float sum = 0;
      for (uint32_t k = 0; k < length; ++k)
      {
        sum += buf[i + k * size_in_ch] * buf[j + k * size_in_ch];
      }
      hessian[i * size_in_ch + j] = 2 * sum;
    }
  }

  HessianVector &vector = _hessian_map[node];
  vector.update(hessian);
}

void HessianComputer::recordHessianForConv2D(const luci::CircleNode *node)
{
  assert(_input_tensor->shape().num_dims() == 4);
  assert(_input_tensor->element_type() == luci_interpreter::DataType::FLOAT32);

  const auto circle_conv2d = loco::must_cast<const luci::CircleConv2D *>(node);
  const auto node_filter = loco::must_cast<luci::CircleConst *>((circle_conv2d)->filter());
  assert(circle_conv2d->rank() >= 4);
  assert(node_filter->dtype() == loco::DataType::FLOAT32);
  assert(node_filter->rank() == 4);

  uint32_t size_in_ch =
    node_filter->size<loco::DataType::FLOAT32>() / circle_conv2d->dim(3).value();

  uint32_t input_n = _input_tensor->shape().dim(0);
  uint32_t input_h = _input_tensor->shape().dim(1);
  uint32_t input_w = _input_tensor->shape().dim(2);
  uint32_t input_c = _input_tensor->shape().dim(3);

  uint32_t stride_h = circle_conv2d->stride()->h();
  uint32_t stride_w = circle_conv2d->stride()->w();
  uint32_t dilation_h = circle_conv2d->dilation()->h();
  uint32_t dilation_w = circle_conv2d->dilation()->w();

  uint32_t kernel_h = node_filter->dim(1).value();
  uint32_t kernel_w = node_filter->dim(2).value();
  uint32_t kernel_ic = node_filter->dim(3).value();

  const auto data = _input_tensor->data<float>();
  const auto num_elements = _input_tensor->shape().num_elements();
  assert(data != 0);
  assert(num_elements != 0);
  std::vector<float> buf(data, data + num_elements);

  unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h, dilation_w,
         kernel_h, kernel_w, kernel_ic);
  assert(size_in_ch != 0);
  uint32_t length = buf.size() / size_in_ch;

  std::vector<float> hessian(size_in_ch * size_in_ch, 0);
  for (uint32_t i = 0; i < size_in_ch; ++i)
  {
    for (uint32_t j = 0; j < size_in_ch; ++j)
    {
      float sum = 0;
      for (uint32_t k = 0; k < length; ++k)
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
    throw std::invalid_argument("RecordHessian: node or input_tensor is null.");

  if (input_tensor->element_type() != luci_interpreter::DataType::FLOAT32)
    throw std::runtime_error("RecordHessian: Unsupported dtype: only FLOAT32 is supported.");

  _input_tensor = input_tensor;

  switch (node->opcode())
  {
    case luci::CircleOpcode::FULLY_CONNECTED:
      recordHessianForFullyConnected(node);
      break;
    case luci::CircleOpcode::CONV_2D:
      recordHessianForConv2D(node);
      break;
    default:
      throw std::runtime_error("RecordHessian: " + node->name() + " is unsupported op.");
  }
}

std::unique_ptr<HessianMap> HessianComputer::getMap()
{
  auto hessian_map = std::make_unique<HessianMap>();

  for (auto item : _hessian_map)
  {
    auto &vec = (*hessian_map)[item.first];
    vec = item.second.hessian;
  }

  return hessian_map;
}

} // namespace record_hessian
