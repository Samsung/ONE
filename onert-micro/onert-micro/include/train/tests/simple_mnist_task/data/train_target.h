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

#ifndef ONERT_MICRO_TRAIN_TESTS_SIMPLE_MNIST_TASK_DATA_TRAIN_TARGET_H
#define ONERT_MICRO_TRAIN_TESTS_SIMPLE_MNIST_TASK_DATA_TRAIN_TARGET_H

#include <vector>
#include <cstring>

namespace onert_micro
{
namespace train
{
namespace test
{
namespace data
{
unsigned char simple_mnist_task_target_data[] = {
  0x00, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02, 0x00, 0x01, 0x01, 0x02, 0x02,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01
};
unsigned int simple_mnist_task_target_data_size = 20;

} // namespace data
} // namespace test
} // namespace train
} // namespace onert_micro

#endif // ONERT_MICRO_TRAIN_TESTS_SIMPLE_MNIST_TASK_DATA_TRAIN_TARGET_H
