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
#include <string>

#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>

namespace
{

// Global variable for error message
std::string last_error_message;

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

// Function to retrieve the last error message.
extern "C" const char *get_last_error(void) { return last_error_message.c_str(); }

// Clear the last error message
extern "C" void clear_last_error(void) { last_error_message.clear(); }

/**
 * @brief A function that wraps another function and catches any exceptions.
 *
 * This function executes the given callable (`func`) with the provided arguments.
 * If the callable throws an exception, the exception message is stored in a
 *  `last_error_message` global variable.
 *
 * @tparam Func The type of the callable funciton.
 * @tparam Args The types of arguments to pass to the callable function.
 * @param func The callable function to execute.
 * @param args The arguments to pass to the callable function.
 * @return The return value of the callable function, or a default value in case of
 *         an exception. If the function has a `void` return type, it simply returns
 *         without any value.
 *
 * @note This function ensures that exceptions are safely caught and conveted to
 *       error messages that can be queried externally, e.g. from Python.
 */
template <typename Func, typename... Args>
auto exception_wrapper(Func func, Args... args) -> typename std::result_of<Func(Args...)>::type
{
  using ReturnType = typename std::result_of<Func(Args...)>::type;

  try
  {
    return func(std::forward<Args>(args)...);
  }
  catch (const std::exception &e)
  {
    last_error_message = e.what();
    if constexpr (not std::is_void<ReturnType>::value)
    {
      return ReturnType{};
    }
  }
  catch (...)
  {
    last_error_message = "Unknown error";
    if constexpr (not std::is_void<ReturnType>::value)
    {
      return ReturnType{};
    }
  }
}

} // namespace

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
    if (_module == nullptr)
    {
      throw std::runtime_error{"Cannot import module."};
    }
    _intp = new luci_interpreter::Interpreter(_module.get());
  }

  ~InterpreterWrapper() { delete _intp; }

  void interpret(void) { _intp->interpret(); }

  void writeInputTensor(const int input_idx, const void *data, size_t input_size)
  {
    const auto input_nodes = loco::input_nodes(_module->graph());
    const auto input_node = loco::must_cast<const luci::CircleInput *>(input_nodes.at(input_idx));
    // Input size from model binary
    const auto fb_input_size = ::getTensorSize(input_node);
    if (fb_input_size != input_size)
    {
      const auto msg = "Invalid input size: " + std::to_string(fb_input_size) +
                       " != " + std::to_string(input_size);
      throw std::runtime_error(msg);
    }
    _intp->writeInputTensor(input_node, data, fb_input_size);
  }

  void readOutputTensor(const int output_idx, void *output, size_t output_size)
  {
    const auto output_nodes = loco::output_nodes(_module->graph());
    const auto output_node =
      loco::must_cast<const luci::CircleOutput *>(output_nodes.at(output_idx));
    const auto fb_output_size = ::getTensorSize(output_node);
    if (fb_output_size != output_size)
    {
      const auto msg = "Invalid output size: " + std::to_string(fb_output_size) +
                       " != " + std::to_string(output_size);
      throw std::runtime_error(msg);
    }
    _intp->readOutputTensor(output_node, output, fb_output_size);
  }

private:
  luci_interpreter::Interpreter *_intp = nullptr;
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
  return ::exception_wrapper([&]() { return new InterpreterWrapper(data, data_size); });
}

void Interpreter_delete(InterpreterWrapper *intp)
{
  ::exception_wrapper([&]() { delete intp; });
}

void Interpreter_interpret(InterpreterWrapper *intp)
{
  ::exception_wrapper([&]() { intp->interpret(); });
}

void Interpreter_writeInputTensor(InterpreterWrapper *intp, const int input_idx, const void *data,
                                  size_t input_size)
{
  ::exception_wrapper([&]() { intp->writeInputTensor(input_idx, data, input_size); });
}

void Interpreter_readOutputTensor(InterpreterWrapper *intp, const int output_idx, void *output,
                                  size_t output_size)
{
  ::exception_wrapper([&]() { intp->readOutputTensor(output_idx, output, output_size); });
}

} // extern "C"
