/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_TRAINING_OPS_RESIZEBILINEAR_H__
#define __ONERT_BACKEND_TRAINING_OPS_RESIZEBILINEAR_H__

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

class ResizeBilinearLayer : public ::onert::exec::IFunction
{
public:
  ResizeBilinearLayer();

public:
  void configure(const IPortableTensor *input1, IPortableTensor *output,
                 const IPortableTensor *size, bool align_corners, bool half_pixel_centers);

  void configure(const IPortableTensor *input, IPortableTensor *output, int32_t output_height,
                 int32_t output_width, bool align_corners, bool half_pixel_centers);

  void run() override;

private:
  const IPortableTensor *_input;
  IPortableTensor *_output;
  const IPortableTensor *_size;
  int32_t _output_height;
  int32_t _output_width;
  bool _align_corners;
  bool _half_pixel_centers;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_RESIZEBILINEAR_H__
