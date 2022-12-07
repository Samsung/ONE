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

#ifndef __ONE_SERVICE_NPUD_TEST_CLIENT_MODEL_H__
#define __ONE_SERVICE_NPUD_TEST_CLIENT_MODEL_H__

#include <stdexcept>
#include <string>
#include <cstring>
#include <libnpuhost.h>

namespace npud
{
namespace tests
{
namespace client
{

class Model
{
public:
  Model(std::string file)
    : _name(file), _meta(nullptr), _inputs({
                                     0,
                                   }),
      _outputs({
        0,
      })
  {
    auto meta = getNPUmodel_metadata(file.c_str(), false);
    if (meta == nullptr)
    {
      std::runtime_error("failed to get model meta data");
    }
    if (NPUBIN_VERSION(meta->magiccode) != 3)
    {
      std::runtime_error("wrong model file");
    }

    _meta = meta;
    init_buffers();
  }
  ~Model() { free(_meta); }

  std::string &get_name() { return _name; }
  input_buffers *get_inputs() { return &_inputs; }
  output_buffers *get_outputs() { return &_outputs; }

private:
  void init_buffers()
  {
    _inputs.num_buffers = _meta->input_seg_num;
    for (auto i = 0; i < _inputs.num_buffers; ++i)
    {
      uint32_t idx = _meta->input_seg_idx[i];
      _inputs.bufs[i].type = BUFFER_MAPPED;
      _inputs.bufs[i].size = _meta->segment_size[idx];
      _inputs.bufs[i].addr = NULL;
    }

    _outputs.num_buffers = _meta->output_seg_num;
    for (auto i = 0; i < _outputs.num_buffers; ++i)
    {
      uint32_t idx = _meta->output_seg_idx[i];
      _outputs.bufs[i].type = BUFFER_MAPPED;
      _outputs.bufs[i].size = _meta->segment_size[idx];
      _outputs.bufs[i].addr = NULL;
    }
  }

private:
  std::string _name;
  npubin_meta *_meta;
  input_buffers _inputs;
  output_buffers _outputs;
};

} // namespace client
} // namespace tests
} // namespace npud

#endif // __ONE_SERVICE_NPUD_TEST_CLIENT_MODEL_H__
