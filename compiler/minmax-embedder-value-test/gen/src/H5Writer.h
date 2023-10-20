/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MINMAX_EMBEDDER_TEST_H5WRITER_H__
#define __MINMAX_EMBEDDER_TEST_H5WRITER_H__

#include "ModelSpec.h"
#include "DataGen.h"

#include <string>

namespace minmax_embedder_test
{
// It must be same to onert/core/src/dumper/h5/MinMaxDumper.h
//
// GROUP /
//   GROUP value
//     └── GROUP run_{idx}
//           └── GROUP model_{idx}
//                 └── GROUP subg_{idx}
//                       ├── DATASET op_{idx}
//                       │      DATATYPE Float32
//                       │      DATASPACE (2)
//                       │      DATA { min, max }
//                       └── DATASET input_{idx}
//                              DATATYPE Float32
//                              DATASPACE (2)
//                              DATA { min, max }
class H5Writer
{
public:
  H5Writer(const ModelSpec &md_spec, const std::string &filepath);
  void dump();

private:
  ModelSpec _md_spec;
  std::string _filepath;
};
} // namespace minmax_embedder_test

#endif // __MINMAX_EMBEDDER_TEST_H5WRITER_H__
