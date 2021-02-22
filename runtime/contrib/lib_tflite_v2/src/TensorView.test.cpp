/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflite/TensorView.h"

#include <cassert>

void int_test(void)
{
  int value[6] = {1, 2, 3, 4, 5, 6};

  const nnfw::misc::tensor::Shape shape{2, 3};
  const nnfw::tflite::TensorView<int> view{shape, value};

  assert(view.at(nnfw::misc::tensor::Index{0, 0}) == 1);
  assert(view.at(nnfw::misc::tensor::Index{0, 1}) == 2);
  assert(view.at(nnfw::misc::tensor::Index{0, 2}) == 3);
  assert(view.at(nnfw::misc::tensor::Index{1, 0}) == 4);
  assert(view.at(nnfw::misc::tensor::Index{1, 1}) == 5);
  assert(view.at(nnfw::misc::tensor::Index{1, 2}) == 6);
}

int main(int argc, char **argv)
{
  float value[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const nnfw::misc::tensor::Shape shape{2, 3};
  const nnfw::tflite::TensorView<float> view{shape, value};

  assert(view.at(nnfw::misc::tensor::Index{0, 0}) == 1.0f);
  assert(view.at(nnfw::misc::tensor::Index{0, 1}) == 2.0f);
  assert(view.at(nnfw::misc::tensor::Index{0, 2}) == 3.0f);
  assert(view.at(nnfw::misc::tensor::Index{1, 0}) == 4.0f);
  assert(view.at(nnfw::misc::tensor::Index{1, 1}) == 5.0f);
  assert(view.at(nnfw::misc::tensor::Index{1, 2}) == 6.0f);

  int_test();

  return 0;
}
