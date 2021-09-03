#!/usr/bin/python3

import re

def print_license(f):
  f.write("/*\n")
  f.write(" * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved\n")
  f.write(" *\n")
  f.write(" * Licensed under the Apache License, Version 2.0 (the \"License\");\n")
  f.write(" * you may not use this file except in compliance with the License.\n")
  f.write(" * You may obtain a copy of the License at\n")
  f.write(" *\n")
  f.write(" *    http://www.apache.org/licenses/LICENSE-2.0\n")
  f.write(" *\n")
  f.write(" * Unless required by applicable law or agreed to in writing, software\n")
  f.write(" * distributed under the License is distributed on an \"AS IS\" BASIS,\n")
  f.write(" * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n")
  f.write(" * See the License for the specific language governing permissions and\n")
  f.write(" * limitations under the License.\n")
  f.write(" */\n")


source_file = None
with open("builders.txt", "r") as f:
  for line in f:
    m = re.match(".*build_kernel_Circle(.*)\(.*$", line)
    if m:
      op_name = m.group(1)
      # create header
      with open("nodes/"+op_name+".h", "w") as header:
        print_license(header)
        header.write("\n")
        header.write("#ifndef LUCI_INTERPRETER_LOADER_NODES_" + op_name.upper() + "_H\n")
        header.write("#define LUCI_INTERPRETER_LOADER_NODES_" + op_name.upper() + "_H\n")
        header.write("\n")
        header.write("#include \"kernels/"+ op_name + ".h\"\n")
        header.write("#include \"loader/KernelBuilderHelper.h\"\n")
        header.write("\n")
        header.write("#include <luci/IR/CircleNodeVisitor.h>\n")
        header.write("\n")
        header.write("namespace luci_interpreter\n")
        header.write("{\n")
        header.write("\n")
        header.write("std::unique_ptr<Kernel> build_kernel_Circle" + op_name + "(const luci::CircleNode *circle_node, KernelBuilderHelper &helper);\n")
        header.write("\n")
        header.write("} // namespace luci_interpreter\n")
        header.write("\n")
        header.write("#endif // LUCI_INTERPRETER_LOADER_NODES_" + op_name.upper() + "_H\n")

      if source_file:
        source_file.write("} // namespace luci_interpreter\n")
        source_file.close()

      # create source
      source_file = open("nodes/" + op_name + ".cpp", "w")
      print_license(source_file)
      source_file.write("\n")
      source_file.write("#include \"" + op_name+".h\"\n")
      source_file.write("\n")
      source_file.write("namespace luci_interpreter\n")
      source_file.write("{\n")
      source_file.write("\n")
      source_file.write("std::unique_ptr<Kernel> build_kernel_Circle" + op_name + "(const luci::CircleNode *circle_node, KernelBuilderHelper &helper)\n")
    else:
      source_file.write(line)

source_file.write("\n")
source_file.write("} // namespace luci_interpreter\n")
source_file.close()

