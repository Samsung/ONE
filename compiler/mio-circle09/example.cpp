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

//
// This example shows how to include and use "mio-circle09"
//
#include <mio/circle/schema_generated.h>

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
  std::ifstream ifs(argv[1], std::ios_base::binary);
  std::vector<char> buf(std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{});

  flatbuffers::Verifier verifier{reinterpret_cast<uint8_t *>(buf.data()), buf.size()};

  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "Fail" << std::endl;
    return 255;
  }

  std::cout << "Pass" << std::endl;
  return 0;
}
