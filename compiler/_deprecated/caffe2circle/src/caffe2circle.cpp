/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <caffe_importer.h>
#include <mir2loco.h>
#include <exo/CircleExporter.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: caffe2circle <path/to/caffe/model> <path/to/circle/model>\n";
    return EXIT_FAILURE;
  }

  const char *caffe_path = argv[1];
  const char *circle_path = argv[2];

  std::unique_ptr<mir::Graph> mir_graph = mir_caffe::importModelFromBinaryFile(caffe_path);
  std::unique_ptr<loco::Graph> loco_graph = mir2loco::Transformer().transform(mir_graph.get());
  exo::CircleExporter(loco_graph.get()).dumpToFile(circle_path);
  return EXIT_SUCCESS;
}
