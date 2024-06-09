/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ARSER_H__
#define __ARSER_H__

#include <iostream>
#include <sstream>

#include <iterator>
#include <typeinfo>

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

#include <cstring>

#include <cassert>
#include <cstdint>

namespace arser
{
namespace internal
{

template <typename T> T lexical_cast(const std::string &str)
{
  std::istringstream ss;
  ss.str(str);
  T data;
  ss >> data;
  return data;
}

template <> inline bool lexical_cast(const std::string &str)
{
  bool data = true;
  if (str == "false" || str == "False" || str == "FALSE" || str == "0")
    data = false;
  return data;
}

template <typename T> inline std::string to_string(const T value) { return std::to_string(value); }

template <> inline std::string to_string(const char *value) { return std::string(value); }

template <> inline std::string to_string(const bool value) { return value ? "true" : "false"; }

/**
 * @brief Returns the string with the leading dash removed.
 *
 * If there is no dash, it returns as it is.
 */
inline std::string remove_dash(const std::string &str)
{
  std::string ret{str};
  auto pos = ret.find_first_not_of('-');
  if (pos == std::string::npos)
    return ret;
  return ret.substr(pos);
}

/**
 * @brief Returns the string that created by concatenating the elements of a vector with commas.
 */
inline std::string make_comma_concatenated(const std::vector<std::string> &vec)
{
  std::ostringstream oss;
  std::copy(vec.begin(), std::prev(vec.end()), std::ostream_iterator<std::string>(oss, ", "));
  oss << vec.back();
  return oss.str();
}

} // namespace internal
} // namespace arser

namespace arser
{

// TypeName declaration
template <typename T> struct TypeName
{
  static const char *Get() { return typeid(T).name(); }
};
template <> struct TypeName<int>
{
  static const char *Get() { return "int"; }
};
template <> struct TypeName<std::vector<int>>
{
  static const char *Get() { return "vector<int>"; }
};
template <> struct TypeName<float>
{
  static const char *Get() { return "float"; }
};
template <> struct TypeName<std::vector<float>>
{
  static const char *Get() { return "vector<float>"; }
};
template <> struct TypeName<bool>
{
  static const char *Get() { return "bool"; }
};
template <> struct TypeName<std::string>
{
  static const char *Get() { return "string"; }
};
template <> struct TypeName<std::vector<std::string>>
{
  static const char *Get() { return "vector<string>"; }
};
template <> struct TypeName<const char *>
{
  static const char *Get() { return "string"; }
};
template <> struct TypeName<std::vector<const char *>>
{
  static const char *Get() { return "vector<string>"; }
};

// supported DataType
enum class DataType
{
  INT32,
  INT32_VEC,
  FLOAT,
  FLOAT_VEC,
  BOOL,
  STR,
  STR_VEC,
};

class Arser;

/**
 * Argument
 *   ├── positional argument
 *   └── optioanl argument  [ dash at the beginning of the string ]
 *       ├── long option    [ two or more dashes ]
 *       └── short option   [ one dash ]
 *
 * Argument has two types - positional argument, optional argument.
 *
 * The way to distinguish the two types is whether there is a dash('-') at the beginning of the
 * string.
 *
 * And, optional argument has two types as well - long option, short option, which is distinguished
 * by the number of dash.
 */
class Argument
{
public:
  explicit Argument(const std::string &arg_name) : _long_name{arg_name}, _names{arg_name} {}
  explicit Argument(const std::string &short_name, const std::string &long_name)
    : _short_name{short_name}, _long_name{long_name}, _names{short_name, long_name}
  {
  }
  explicit Argument(const std::string &short_name, const std::string &long_name,
                    const std::vector<std::string> &names)
    : _short_name{short_name}, _long_name{long_name}, _names{names}
  {
    // 'names' must have 'short_name' and 'long_name'.
    auto it = std::find(names.begin(), names.end(), short_name);
    assert(it != names.end());
    it = std::find(names.begin(), names.end(), long_name);
    assert(it != names.end());
    // for avoiding unused warning.
    (void)it;
  }

  Argument &nargs(uint32_t num)
  {
    if (num == 0)
    {
      _type = "bool";
    }
    _nargs = num;
    return *this;
  }

  Argument &type(DataType type)
  {
    switch (type)
    {
      case DataType::INT32:
        _type = "int";
        break;
      case DataType::INT32_VEC:
        _type = "vector<int>";
        break;
      case DataType::FLOAT:
        _type = "float";
        break;
      case DataType::FLOAT_VEC:
        _type = "vector<float>";
        break;
      case DataType::BOOL:
        _type = "bool";
        break;
      case DataType::STR:
        _type = "string";
        break;
      case DataType::STR_VEC:
        _type = "vector<string>";
        break;
      default:
        throw std::runtime_error("NYI DataType");
    }
    return *this;
  }

  Argument &required(void)
  {
    _is_required = true;
    return *this;
  }

  Argument &required(bool value)
  {
    _is_required = value;
    return *this;
  }

  Argument &accumulated(void)
  {
    _is_accumulated = true;
    return *this;
  }

  Argument &accumulated(bool value)
  {
    _is_accumulated = value;
    return *this;
  }

  Argument &help(std::string help_message)
  {
    _help_message = help_message;
    return *this;
  }

  Argument &exit_with(const std::function<void(void)> &func)
  {
    _func = func;
    return *this;
  }

  template <typename T> Argument &default_value(const T value)
  {
    if ((_nargs <= 1 && TypeName<T>::Get() == _type) ||
        (_nargs > 1 && TypeName<std::vector<T>>::Get() == _type))
      _values.emplace_back(internal::to_string(value));
    else
    {
      throw std::runtime_error("Type mismatch. "
                               "You called default_value() method with a type different "
                               "from the one you specified. "
                               "Please check the type of what you specified in "
                               "add_argument() method.");
    }
    return *this;
  }

  template <typename T, typename... Ts> Argument &default_value(const T value, const Ts... values)
  {
    if ((_nargs <= 1 && TypeName<T>::Get() == _type) ||
        (_nargs > 1 && TypeName<std::vector<T>>::Get() == _type))
    {
      _values.emplace_back(internal::to_string(value));
      default_value(values...);
    }
    else
    {
      throw std::runtime_error("Type mismatch. "
                               "You called default_value() method with a type different "
                               "from the one you specified. "
                               "Please check the type of what you specified in "
                               "add_argument() method.");
    }
    return *this;
  }

private:
  // The '_names' vector contains all of the options specified by the user.
  // And among them, '_long_name' and '_short_name' are selected.
  std::string _long_name;
  std::string _short_name;
  std::vector<std::string> _names;
  std::string _type = "string";
  std::string _help_message;
  std::function<void(void)> _func;
  uint32_t _nargs{1};
  bool _is_required{false};
  bool _is_accumulated{false};
  std::vector<std::string> _values;
  std::vector<std::vector<std::string>> _accum_values;

  friend class Arser;
  friend std::ostream &operator<<(std::ostream &, const Arser &);
};

class Arser
{
public:
  explicit Arser(const std::string &program_description = {})
    : _program_description{program_description}
  {
    add_argument("-h", "--help").help("Show help message and exit").nargs(0);
  }

  Argument &add_argument(const std::string &arg_name)
  {
    if (arg_name.at(0) != '-') /* positional */
    {
      _positional_arg_vec.emplace_back(arg_name);
      _arg_map[arg_name] = &_positional_arg_vec.back();
    }
    else /* optional */
    {
      // The length of optional argument name must be 2 or more.
      // And it shouldn't be hard to recognize. e.g. '-', '--'
      if (arg_name.size() < 2)
      {
        throw std::runtime_error("Too short name. The length of argument name must be 2 or more.");
      }
      if (arg_name == "--")
      {
        throw std::runtime_error(
          "Too short name. Option name must contain at least one character other than dash.");
      }
      _optional_arg_vec.emplace_back(arg_name);
      _optional_arg_vec.back()._short_name = arg_name;
      _arg_map[arg_name] = &_optional_arg_vec.back();
    }
    return *_arg_map[arg_name];
  }

  Argument &add_argument(const std::vector<std::string> &arg_name_vec)
  {
    assert(arg_name_vec.size() >= 2);
    std::string long_opt, short_opt;
    // find long and short option
    for (const auto &arg_name : arg_name_vec)
    {
      if (arg_name.at(0) != '-')
      {
        throw std::runtime_error("Invalid argument. "
                                 "Positional argument cannot have short option.");
      }
      assert(arg_name.size() >= 2);
      if (long_opt.empty() && arg_name.at(0) == '-' && arg_name.at(1) == '-')
      {
        long_opt = arg_name;
      }
      if (short_opt.empty() && arg_name.at(0) == '-' && arg_name.at(1) != '-')
      {
        short_opt = arg_name;
      }
    }
    // If one of the two is empty, fill it with the non-empty one for pretty printing.
    if (long_opt.empty())
    {
      assert(not short_opt.empty());
      long_opt = short_opt;
    }
    if (short_opt.empty())
    {
      assert(not long_opt.empty());
      short_opt = long_opt;
    }

    _optional_arg_vec.emplace_back(short_opt, long_opt, arg_name_vec);
    for (const auto &arg_name : arg_name_vec)
    {
      _arg_map[arg_name] = &_optional_arg_vec.back();
    }
    return _optional_arg_vec.back();
  }

  template <typename... Ts> Argument &add_argument(const std::string &arg_name, Ts... arg_names)
  {
    if (sizeof...(arg_names) == 0)
    {
      return add_argument(arg_name);
    }
    // sizeof...(arg_names) > 0
    else
    {
      return add_argument(std::vector<std::string>{arg_name, arg_names...});
    }
  }

  void validate_arguments(void)
  {
    // positional argument is always required.
    for (const auto &arg : _positional_arg_vec)
    {
      if (arg._is_required)
      {
        throw std::runtime_error("Invalid arguments. Positional argument must always be required.");
      }
    }
    // TODO accumulated arguments shouldn't be enabled to positional arguments.
    // TODO accumulated arguments shouldn't be enabled to optional arguments whose `narg` == 0.
  }

  void parse(int argc, char **argv)
  {
    validate_arguments();
    _program_name = argv[0];
    _program_name.erase(0, _program_name.find_last_of("/\\") + 1);
    if (argc >= 2)
    {
      if (!std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h"))
      {
        std::cout << *this;
        std::exit(0);
      }
      else
      {
        for (const auto &arg : _arg_map)
        {
          const auto &func = arg.second->_func;
          if (func && !std::strcmp(argv[1], arg.first.c_str()))
          {
            func();
            std::exit(0);
          }
        }
      }
    }
    /*
    ** ./program_name [optional argument] [positional argument]
    */
    // get the number of positioanl argument
    size_t parg_num = _positional_arg_vec.size();
    // get the number of "required" optional argument
    size_t required_oarg_num = 0;
    for (auto arg : _optional_arg_vec)
    {
      if (arg._is_required)
        required_oarg_num++;
    }
    // parse argument
    for (int c = 1; c < argc;)
    {
      std::string arg_name{argv[c++]};
      auto arg = _arg_map.find(arg_name);
      // check whether arg is positional or not
      if (arg == _arg_map.end())
      {
        if (parg_num)
        {
          auto it = _positional_arg_vec.begin();
          std::advance(it, _positional_arg_vec.size() - parg_num);
          (*it)._values.clear();
          (*it)._values.emplace_back(arg_name);
          parg_num--;
        }
        else
          throw std::runtime_error("Invalid argument. "
                                   "You've given more positional argument than necessary.");
      }
      else // optional argument
      {
        // check whether arg is required or not
        if (arg->second->_is_required)
          required_oarg_num--;
        arg->second->_values.clear();
        for (uint32_t n = 0; n < arg->second->_nargs; n++)
        {
          if (c >= argc)
            throw std::runtime_error("Invalid argument. "
                                     "You must have missed some argument.");
          arg->second->_values.emplace_back(argv[c++]);
        }
        // accumulate values
        if (arg->second->_is_accumulated)
        {
          arg->second->_accum_values.emplace_back(arg->second->_values);
        }
        if (arg->second->_nargs == 0)
        {
          // TODO std::boolalpha for true or false
          arg->second->_values.emplace_back("1");
        }
      }
    }
    if (parg_num || required_oarg_num)
      throw std::runtime_error("Invalid argument. "
                               "You must have missed some argument.");
  }

  bool operator[](const std::string &arg_name)
  {
    auto arg = _arg_map.find(arg_name);
    if (arg == _arg_map.end())
      return false;

    if (arg->second->_is_accumulated)
      return arg->second->_accum_values.size() > 0 ? true : false;

    return arg->second->_values.size() > 0 ? true : false;
  }

  template <typename T> T get_impl(const std::string &arg_name, T *);

  template <typename T> std::vector<T> get_impl(const std::string &arg_name, std::vector<T> *);

  template <typename T>
  std::vector<std::vector<T>> get_impl(const std::string &arg_name, std::vector<std::vector<T>> *);

  template <typename T> T get(const std::string &arg_name);

  friend std::ostream &operator<<(std::ostream &stream, const Arser &parser)
  {
    // print description
    if (!parser._program_description.empty())
    {
      stream << "What " << parser._program_name << " does: " << parser._program_description
             << "\n\n";
    }
    /*
    ** print usage
    */
    auto print_usage_arg = [&](const arser::Argument &arg) {
      stream << " ";
      std::string arg_name = arser::internal::remove_dash(arg._long_name);
      std::for_each(arg_name.begin(), arg_name.end(),
                    [&stream](const char &c) { stream << static_cast<char>(::toupper(c)); });
    };
    stream << "Usage: ./" << parser._program_name << " ";
    // required optional argument
    for (const auto &arg : parser._optional_arg_vec)
    {
      if (!arg._is_required)
        continue;
      stream << arg._short_name;
      print_usage_arg(arg);
      stream << " ";
    }
    // rest of the optional argument
    for (const auto &arg : parser._optional_arg_vec)
    {
      if (arg._is_required)
        continue;
      stream << "[" << arg._short_name;
      if (arg._nargs)
      {
        print_usage_arg(arg);
      }
      stream << "]"
             << " ";
    }
    // positional arguement
    for (const auto &arg : parser._positional_arg_vec)
    {
      stream << arg._long_name << " ";
    }
    stream << "\n\n";
    /*
    ** print argument list and its help message
    */
    // get the length of the longest argument
    size_t length_of_longest_arg = 0;
    for (const auto &arg : parser._positional_arg_vec)
    {
      length_of_longest_arg = std::max(length_of_longest_arg,
                                       arser::internal::make_comma_concatenated(arg._names).size());
    }
    for (const auto &arg : parser._optional_arg_vec)
    {
      length_of_longest_arg = std::max(length_of_longest_arg,
                                       arser::internal::make_comma_concatenated(arg._names).size());
    }

    const size_t message_width = 60;
    auto print_help_args = [&](const std::list<Argument> &args, const std::string &title) {
      if (!args.empty())
      {
        stream << title << std::endl;
        for (const auto &arg : args)
        {
          stream.width(length_of_longest_arg);
          stream << std::left << arser::internal::make_comma_concatenated(arg._names) << "\t";
          for (size_t i = 0; i < arg._help_message.length(); i += message_width)
          {
            if (i)
              stream << std::string(length_of_longest_arg, ' ') << "\t";
            stream << arg._help_message.substr(i, message_width) << std::endl;
          }
        }
        std::cout << std::endl;
      }
    };
    // positional argument
    print_help_args(parser._positional_arg_vec, "[Positional argument]");
    // optional argument
    print_help_args(parser._optional_arg_vec, "[Optional argument]");

    return stream;
  }

private:
  std::string _program_name;
  std::string _program_description;
  std::list<Argument> _positional_arg_vec;
  std::list<Argument> _optional_arg_vec;
  std::map<std::string, Argument *> _arg_map;
};

template <typename T> T Arser::get_impl(const std::string &arg_name, T *)
{
  auto arg = _arg_map.find(arg_name);
  if (arg == _arg_map.end())
    throw std::runtime_error("Invalid argument. "
                             "There is no argument you are looking for: " +
                             arg_name);

  if (arg->second->_is_accumulated)
    throw std::runtime_error(
      "Type mismatch. "
      "You called get using a type different from the one you specified."
      "Accumulated argument is returned as std::vector of the specified type");

  if (arg->second->_type != TypeName<T>::Get())
    throw std::runtime_error("Type mismatch. "
                             "You called get() method with a type different "
                             "from the one you specified. "
                             "Please check the type of what you specified in "
                             "add_argument() method.");

  if (arg->second->_values.size() == 0)
    throw std::runtime_error("Wrong access. "
                             "You must make sure that the argument is given before accessing it. "
                             "You can do it by calling arser[\"argument\"].");

  return internal::lexical_cast<T>(arg->second->_values[0]);
}

template <typename T> std::vector<T> Arser::get_impl(const std::string &arg_name, std::vector<T> *)
{
  auto arg = _arg_map.find(arg_name);
  if (arg == _arg_map.end())
    throw std::runtime_error("Invalid argument. "
                             "There is no argument you are looking for: " +
                             arg_name);

  // Accumulated arguments with scalar type (e.g., STR)
  if (arg->second->_is_accumulated)
  {
    if (arg->second->_type != TypeName<T>::Get())
      throw std::runtime_error("Type mismatch. "
                               "You called get using a type different from the one you specified.");

    std::vector<T> data;
    for (auto values : arg->second->_accum_values)
    {
      assert(values.size() == 1);
      data.emplace_back(internal::lexical_cast<T>(values[0]));
    }
    return data;
  }

  if (arg->second->_type != TypeName<std::vector<T>>::Get())
    throw std::runtime_error("Type mismatch. "
                             "You called get using a type different from the one you specified.");

  std::vector<T> data;
  std::transform(arg->second->_values.begin(), arg->second->_values.end(), std::back_inserter(data),
                 [](std::string str) -> T { return internal::lexical_cast<T>(str); });
  return data;
}

// Accumulated arguments with vector type (e.g., STR_VEC)
template <typename T>
std::vector<std::vector<T>> Arser::get_impl(const std::string &arg_name,
                                            std::vector<std::vector<T>> *)
{
  auto arg = _arg_map.find(arg_name);
  if (arg == _arg_map.end())
    throw std::runtime_error("Invalid argument. "
                             "There is no argument you are looking for: " +
                             arg_name);

  if (not arg->second->_is_accumulated)
    throw std::runtime_error("Type mismatch. "
                             "You called get using a type different from the one you specified.");

  if (arg->second->_type != TypeName<std::vector<T>>::Get())
    throw std::runtime_error(
      "Type mismatch. "
      "You called get using a type different from the one you specified."
      "Accumulated argument is returned as std::vector of the specified type");

  std::vector<std::vector<T>> result;
  for (auto values : arg->second->_accum_values)
  {
    std::vector<T> data;
    std::transform(values.begin(), values.end(), std::back_inserter(data),
                   [](std::string str) -> T { return internal::lexical_cast<T>(str); });
    result.emplace_back(data);
  }

  return result;
}

template <typename T> T Arser::get(const std::string &arg_name)
{
  return get_impl(arg_name, static_cast<T *>(nullptr));
}

class Helper
{
public:
  static void add_version(Arser &arser, const std::function<void(void)> &func)
  {
    arser.add_argument("--version")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("Show version information and exit")
      .exit_with(func);
  }

  static void add_verbose(Arser &arser)
  {
    arser.add_argument("-V", "--verbose")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("output additional information to stdout or stderr");
  }
};

} // namespace arser

#endif // __ARSER_H__
