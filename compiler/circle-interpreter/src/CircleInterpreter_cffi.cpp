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

/*
 * This library is for using luci-interpreter in Python. It is implemented
 *  with CFFI. CFFI is a FFI (Foreign Function Interface) for Python calling
 *  C code.
 */

#include <cstddef>

#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/*
 * Q) Why do we need this wrapper class?
 *
 * A) This class is designed to address specific constraints introduced by
 *   the use of CFFI for Python bindings. The original class lacked the
 *   ability to maintain internal state like luci::Module because it is defined
 *   out of the class. But, in Python, there's no way to keep luci::Module in memory.
 *   To overcome this limitation, a wrapper class is implemented to manage and
 *   keep states efficiently.
 *
 *   Moreover, the original interface relied on passing C++ classes as arguments,
 *   which posed compatibility challenges when exposed to Python via CFFI. To
 *   simplify the integration process, the wrapper class redesigns the interface
 *   to accept more generic inputs, such as primitive types or standard containers,
 *   ensuring seamless interaction between C++ and Python.
 *
 *   Overall, the redesigned class or the wrapper class preserves the original
 *   functionality while introducing state management and a more flexible inteface,
 *   making it highly suitable for Python-C++ interoperability through CFFI.
 */
class InterpreterWrapper
{
public:
  explicit InterpreterWrapper(const uint8_t *data, const size_t data_size)
  {
    luci::Importer importer;
    _module = importer.importModule(data, data_size);
    _intp = new luci_interpreter::Interpreter(_module.get());
  }

  ~InterpreterWrapper() { delete _intp; }

  void interpret(void) { _intp->interpret(); }

  void writeInputTensor(const int input_idx, const void *data)
  {
    const auto input_nodes = loco::input_nodes(_module->graph());
    const auto target_input = loco::must_cast<const luci::CircleInput *>(input_nodes.at(input_idx));
    _intp->writeInputTensor(target_input, data, ::getTensorSize(target_input));
  }

  void readOutputTensor(const int output_idx, void *output, size_t output_size)
  {
    const auto output_nodes = loco::output_nodes(_module->graph());
    const auto output_node =
      loco::must_cast<const luci::CircleOutput *>(output_nodes.at(output_idx));
    _intp->readOutputTensor(output_node, output, output_size);
  }

private:
  luci_interpreter::Interpreter *_intp;
  std::unique_ptr<luci::Module> _module;
};

/*
 * CFFI primarily uses functions instead of classes because it is designed to
 *  work with C-compatible interfaces.
 *
 * - This extern "C" is necessary to avoid name mangling.
 * - Explicitly pass the object pointer to any funcitons that operates on the object.
 */
extern "C" {
InterpreterWrapper *Interpreter_new(const uint8_t *data, const size_t data_size)
{
  return new InterpreterWrapper(data, data_size);
}

void Interpreter_delete(InterpreterWrapper *intp) { delete intp; }

void Interpreter_interpret(InterpreterWrapper *intp) { intp->interpret(); }

void Interpreter_writeInputTensor(InterpreterWrapper *intp, const int input_idx, const void *data)
{
  intp->writeInputTensor(input_idx, data);
}

void Interpreter_readOutputTensor(InterpreterWrapper *intp, const int output_idx, void *output,
                                  size_t output_size)
{
  intp->readOutputTensor(output_idx, output, output_size);
}
}
