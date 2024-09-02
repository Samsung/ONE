/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SerializedData.h"

namespace luci
{

void CircleExportMetadata::clear(void)
{
  _source_table.clear();
  _op_table.clear();
  _execution_plan_table.clear();
}

void SerializedModelData::clear(void)
{
  _operator_codes.clear();
  _buffers.clear();
  _metadata.clear();
  _cached_buffer_id.clear();

  // clear extended buffer mode
  _ext_buffer = false;
  _require_ext_buffer = false;
  _buffer_data_map.clear();
}

} // namespace luci
