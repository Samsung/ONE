/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BulkLayer.h"

#include "../Convert.h"

namespace onert::backend::trix::ops
{

BulkLayer::BulkLayer() : _inputs(), _outputs(), _model_id(0), _dev_context(nullptr)
{
  // DO NOTHING
}

BulkLayer::~BulkLayer() { _dev_context->unRegisterModel(_model_id); }

void BulkLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                          std::vector<IPortableTensor *> &outputs, std::string binary_path,
                          const std::shared_ptr<DevContext> &dev_context)
{
  _inputs = inputs;
  _outputs = outputs;
  _dev_context = dev_context;
  _model_id = _dev_context->registerModel(binary_path);
}

void BulkLayer::run()
{
  tensors_data_info in_info;
  tensors_data_info out_info;
  setDataInfo(_inputs, &in_info);
  setDataInfo(_outputs, &out_info);

  input_buffers input_bufs;
  output_buffers output_bufs;
  setBuffers(_inputs, &input_bufs);
  setBuffers(_outputs, &output_bufs);

  size_t batch_size = 1;
  // TODO Remove this assumption
  if (_inputs.size() == 1 && _outputs.size() == 1 && _inputs.at(0)->getShape().dim(0) > 1)
  {
    batch_size = _inputs.at(0)->getShape().dim(0);
  }
  _dev_context->requestRun(_model_id, &input_bufs, &in_info, &output_bufs, &out_info, batch_size);
}

void BulkLayer::prepare()
{
  // DO NOTHING
}

} // namespace onert::backend::trix::ops
