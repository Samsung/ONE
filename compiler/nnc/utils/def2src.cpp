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

#include <iostream>
#include <fstream>
#include <cassert>

int fileToArray(const std::string &source, const std::string &dest, const std::string &arrName)
{
  FILE *fs = fopen(source.c_str(), "rb");
  if (!fs)
  {
    std::cerr << "source file not found: <" << source << ">" << std::endl;
    return -1;
  }

  std::ofstream fo(dest.c_str());
  if (fo.fail())
  {
    std::cerr << "cannot generate file: <" << dest << ">" << std::endl;
    fclose(fs);
    return -1;
  }

  std::cout << "generating <" << dest << ">" << std::endl;

  fo << "#ifndef _" << arrName << "_H_" << std::endl;
  fo << "#define _" << arrName << "_H_" << std::endl;

  fo << "const char " << arrName << "[] = {" << std::endl;

  int is_error = fseek(fs, 0L, SEEK_SET);
  assert(!is_error);
  (void)is_error;
  size_t bytes;
  do
  {
    char buf[1024];
    bytes = fread(buf, 1, sizeof(buf), fs);
    assert(!ferror(fs) && "file read error");

    // convert line
    for (size_t i = 0; i < bytes; i++)
    {
      fo << "0x" << std::hex << static_cast<int>(buf[i]) << ", ";
    }
  } while (bytes != 0);

  fo << "};" << std::endl;

  fo << std::endl;
  fo << "#endif /* _" << arrName << "_H_ */" << std::endl;

  fo.flush();
  fclose(fs);

  return 0;
}

std::string extractFileName(std::string path)
{
  auto pos = path.find_last_of('/');
  if (pos != std::string::npos)
    path = path.substr(pos + 1);

  pos = path.find_first_of('.');
  if (pos != std::string::npos)
    path = path.substr(0, pos);

  return path;
}

int main(int argc, char *argv[])
{
  if (argc < 3)
    return -1;

  std::string OutPutDir = argv[1];

  for (int i = 2; i < argc; i++)
  {
    std::string sourceFullFileName = argv[i];
    std::string filename = extractFileName(sourceFullFileName);
    // NOLINTNEXTLINE (performance-inefficient-string-concatenation)
    std::string outputFileName = OutPutDir + "/" + filename + ".generated.h";

    if (fileToArray(sourceFullFileName, outputFileName, filename) != 0)
      return -1;
  }

  return 0;
}
