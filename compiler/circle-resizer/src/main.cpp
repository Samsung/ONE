/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleResizer.h"
#include "Shape.h"

using namespace circle_resizer;

int main(int argc, char *argv[]) {
  CircleResizer resizer(argv[1]);
  resizer.resize_model({Shape{Dim{1}, Dim{3}}}); // experiment
  resizer.save_model(argv[2]);
  return 0;
}
