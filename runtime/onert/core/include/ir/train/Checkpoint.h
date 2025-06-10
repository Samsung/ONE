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

#ifndef __ONERT_IR_TRAIN_CHECKPOINT_H__
#define __ONERT_IR_TRAIN_CHECKPOINT_H__

namespace onert::ir::train::checkpoint
{

struct __attribute__((packed)) Header
{
  uint16_t magic;
  uint8_t schema;
  uint8_t reserved;
  uint32_t opt1_offset;
  uint32_t opt2_offset;
  uint32_t other_offset;
  uint32_t length;
};

struct __attribute__((packed)) Footer
{
  uint32_t cur_step;
  uint32_t cur_epoch;
};

constexpr uint16_t MAGIC_NUMBER = 429;
constexpr uint8_t SCHEMA_VERSION = 1;

} // namespace onert::ir::train::checkpoint

#endif // __ONERT_IR_TRAIN_CHECKPOINT_H__
