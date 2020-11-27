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

namespace
{

template <typename T> T lexical_cast(const std::string &str)
{
  std::istringstream ss;
  ss.str(str);
  T data;
  ss >> data;
  return data;
}

template <> bool lexical_cast(const std::string &str)
{
  bool data = true;
  if (str == "false" || str == "False" || str == "FALSE" || str == "0")
    data = false;
  return data;
}

template <typename T> inline std::string to_string(const T value) { return std::to_string(value); }

template <> inline std::string to_string(const char *value) { return std::string(value); }

template <> inline std::string to_string(const bool value) { return value ? "true" : "false"; }

} // namespace

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

class Argument
{
public:
  explicit Argument(const std::string &arg_name) : _name{arg_name} {}

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
      _values.emplace_back(::to_string(value));
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
      _values.emplace_back(::to_string(value));
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
  std::string _name;
  std::string _type;
  std::string _help_message;
  std::function<void(void)> _func;
  uint32_t _nargs{1};
  bool _is_required{false};
  std::vector<std::string> _values;

  friend class Arser;
  friend std::ostream &operator<<(std::ostream &, const Arser &);
};

class Arser
{
public:
  explicit Arser(const std::string &program_description = {})
    : _program_description{program_description}
  {
    add_argument("--help").help("Show help message and exit").nargs(0);
  }

  Argument &add_argument(const std::string &arg_name)
  {
    if (arg_name.at(0) != '-')
    {
      _positional_arg_vec.emplace_back(arg_name);
      _arg_map[arg_name] = &_positional_arg_vec.back();
    }
    else
    {
      _optional_arg_vec.emplace_back(arg_name);
      _arg_map[arg_name] = &_optional_arg_vec.back();
    }
    return *_arg_map[arg_name];
  }

  void parse(int argc, char **argv)
  {
    _program_name = argv[0];
    _program_name.erase(0, _program_name.find_last_of("/\\") + 1);
    if (argc >= 2)
    {
      if (!std::strcmp(argv[1], "--help"))
      {
        std::cout << *this;
        std::exit(0);
      }
      else
      {
        for (const auto &arg : _arg_map)
        {
          const auto &func = arg.second->_func;
          if (func && !std::strcmp(argv[1], arg.second->_name.c_str()))
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

    return arg->second->_values.size() > 0 ? true : false;
  }

  template <typename T> T get_impl(const std::string &arg_name, T *);

  template <typename T> std::vector<T> get_impl(const std::string &arg_name, std::vector<T> *);

  template <typename T> T get(const std::string &arg_name);

private:
  std::string _program_name;
  std::string _program_description;
  std::list<Argument> _positional_arg_vec;
  std::list<Argument> _optional_arg_vec;
  std::map<std::string, Argument *> _arg_map;

  friend std::ostream &operator<<(std::ostream &, const Arser &);
};

template <typename T> T Arser::get_impl(const std::string &arg_name, T *)
{
  auto arg = _arg_map.find(arg_name);
  if (arg == _arg_map.end())
    throw std::runtime_error("Invalid argument. "
                             "There is no argument you are looking for.");

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

  return ::lexical_cast<T>(arg->second->_values[0]);
}

template <typename T> std::vector<T> Arser::get_impl(const std::string &arg_name, std::vector<T> *)
{
  auto arg = _arg_map.find(arg_name);
  if (arg == _arg_map.end())
    throw std::runtime_error("Invalid argument. "
                             "There is no argument you are looking for.");

  if (arg->second->_type != TypeName<std::vector<T>>::Get())
    throw std::runtime_error("Type mismatch. "
                             "You called get using a type different from the one you specified.");

  std::vector<T> data;
  std::transform(arg->second->_values.begin(), arg->second->_values.end(), std::back_inserter(data),
                 [](std::string str) -> T { return ::lexical_cast<T>(str); });
  return data;
}

template <typename T> T Arser::get(const std::string &arg_name)
{
  return get_impl(arg_name, static_cast<T *>(nullptr));
}

std::ostream &operator<<(std::ostream &stream, const Arser &parser)
{
  // print description
  if (!parser._program_description.empty())
  {
    stream << "What " << parser._program_name << " does: " << parser._program_description << "\n\n";
  }
  /*
  ** print usage
  */
  stream << "Usage: ./" << parser._program_name << " ";
  // required optional argument
  for (const auto &arg : parser._optional_arg_vec)
  {
    if (!arg._is_required)
      continue;
    stream << arg._name << " ";
    std::string arg_name = arg._name.substr(2);
    std::for_each(arg_name.begin(), arg_name.end(),
                  [&stream](const char &c) { stream << static_cast<char>(::toupper(c)); });
    stream << " ";
  }
  // rest of the optional argument
  for (const auto &arg : parser._optional_arg_vec)
  {
    if (arg._is_required)
      continue;
    stream << "[" << arg._name;
    if (arg._nargs)
    {
      stream << " ";
      std::string arg_name = arg._name.substr(2);
      std::for_each(arg_name.begin(), arg_name.end(),
                    [&stream](const char &c) { stream << static_cast<char>(::toupper(c)); });
    }
    stream << "]"
           << " ";
  }
  // positional arguement
  for (const auto &arg : parser._positional_arg_vec)
  {
    stream << arg._name << " ";
  }
  stream << "\n\n";
  /*
  ** print argument list and its help message
  */
  // get the length of the longest argument
  size_t length_of_longest_arg = 0;
  for (const auto &arg : parser._positional_arg_vec)
  {
    length_of_longest_arg = std::max(length_of_longest_arg, arg._name.length());
  }
  for (const auto &arg : parser._optional_arg_vec)
  {
    length_of_longest_arg = std::max(length_of_longest_arg, arg._name.length());
  }

  const size_t message_width = 60;
  // positional argument
  if (!parser._positional_arg_vec.empty())
  {
    stream << "[Positional argument]" << std::endl;
    for (const auto &arg : parser._positional_arg_vec)
    {
      stream.width(length_of_longest_arg);
      stream << std::left << arg._name << "\t";
      for (size_t i = 0; i < arg._help_message.length(); i += message_width)
      {
        if (i)
          stream << std::string(length_of_longest_arg, ' ') << "\t";
        stream << arg._help_message.substr(i, message_width) << std::endl;
      }
    }
    std::cout << std::endl;
  }
  // optional argument
  if (!parser._optional_arg_vec.empty())
  {
    stream << "[Optional argument]" << std::endl;
    for (const auto &arg : parser._optional_arg_vec)
    {
      stream.width(length_of_longest_arg);
      stream << std::left << arg._name << "\t";
      for (size_t i = 0; i < arg._help_message.length(); i += message_width)
      {
        if (i)
          stream << std::string(length_of_longest_arg, ' ') << "\t";
        stream << arg._help_message.substr(i, message_width) << std::endl;
      }
    }
  }

  return stream;
}

} // namespace arser
