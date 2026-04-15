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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <type_traits>
#include "cstring"

#include "support/CommandLine.h"

namespace nnc
{
namespace cli
{

constexpr const char *const IOption::_groupNames[];

static std::vector<std::string> splitByComma(const char *str)
{
  const char *cur_str = str;
  std::vector<std::string> ret;

  if (std::string(str).empty())
    return ret;

  for (size_t i = 0, cnt = 0; str[i] != '\0'; i++)
  {
    if (str[i] == ',')
    {
      std::string name(cur_str, cnt);
      name.erase(remove_if(name.begin(), name.end(), isspace), name.end());
      cnt = 0;

      ret.push_back(name);

      cur_str = &str[i + 1];
      continue;
    }

    cnt++;
  }

  // push string after last comma
  std::string name(cur_str);
  name.erase(remove_if(name.begin(), name.end(), isspace), name.end());
  ret.push_back(name);

  return ret;

} // splitByComma

std::vector<std::string> optname(const char *names) { return splitByComma(names); }

std::vector<std::string> optvalues(const char *vals) { return splitByComma(vals); }

std::vector<char> separators(const char *seps)
{
  std::vector<char> ret;
  int i;

  if (std::string(seps).empty())
    return ret;

  for (i = 0; isspace(seps[i]); i++)
    ;

  if (seps[i])
  {
    ret.push_back(seps[i]);
    i++;
  }

  for (; seps[i] != '\0'; i++)
  {
    if (seps[i] == ',')
    {
      for (i++; isspace(seps[i]); i++)
        ;

      ret.push_back(seps[i]);
    }
  }

  return ret;
}

CommandLine *CommandLine::getParser()
{
  static CommandLine Parser;

  return &Parser;

} // getParser

/**
 * @param options - vector of all options
 * @return maximum name length of size among all options
 */
static size_t calcMaxLenOfOptionsNames(std::vector<IOption *> options)
{
  size_t max_len = 0, len;

  for (const auto opt : options)
    if (!opt->isDisabled())
    {
      len = 0;
      for (const auto &n : opt->getNames())
        len += n.length();
      max_len = (max_len < len) ? len : max_len;
    }

  return max_len;

} // calcMaxLenOfOptionsNames

/**
 * @brief print option in help message
 * @param opt - option that will be printed
 * @param max_opt_name_len - maximum name length of size among all options
 * @param leading_spaces - leading spaces that will be printed before option name
 */
static void printOption(IOption *opt, size_t max_opt_name_len, size_t leading_spaces)
{

  const auto &option_descr = opt->getOverview();
  const auto &names = opt->getNames();

  std::string option_names(names[0]); // initialize with option name

  // add option aliases to option_names and count them length
  for (size_t i = 1; i < names.size(); i++)
    option_names += ", " + names[i];

  std::string spaces(max_opt_name_len - option_names.length() + leading_spaces, ' ');
  std::cerr << "        " << option_names << spaces << "-    " << option_descr << std::endl;

} // printOption

[[noreturn]] void CommandLine::usage(const std::string &msg, int exit_code)
{
  if (!msg.empty())
  {
    std::cerr << msg << "\n";
  }

  std::cerr << "Usage: " << _prog_name << " OPTIONS\n";
  std::cerr << "Available OPTIONS" << std::endl;

  // determine max length
  size_t max_len = calcMaxLenOfOptionsNames(_options);

  for (const auto opt : _options)
  {
    if (opt->isDisabled())
      // options that are disabled not have to be shown
      continue;

    if (opt->isGrouped())
      // options, that are grouped, will be printed later
      continue;

    printOption(opt, max_len, 4);
  }

  // print grouped options
  for (const auto &group : _grouped_options)
  {
    std::cerr << "Options from '" << group.second[0]->getGroupName() << "' group:" << std::endl;

    for (const auto opt : group.second)
    {
      printOption(opt, max_len, 4);
    }
  }

  exit(exit_code);

} // usage

void CommandLine::registerOption(IOption *opt)
{
  for (const auto &n : opt->getNames())
  {
    auto i = _options_name.emplace(n, opt);

    if (!i.second)
    {
      std::cerr << "option name must be unique: `" << n << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  _options.push_back(opt);

  if (opt->isGrouped())
  {
    auto it = _grouped_options.find(opt->getGroup());

    if (it == _grouped_options.end())
      _grouped_options.emplace(opt->getGroup(), std::vector<IOption *>{opt});
    else
      it->second.push_back(opt);
  }

} // registerOption

IOption *CommandLine::findOption(const char *optname)
{
  auto it = _options_name.find(optname);

  if (it == _options_name.end())
  {
    // optname can contain separators, try
    // to strip these separators and repeat a search
    size_t i = 0;
    for (; optname[i] != '\0' && optname[i] != '=' && optname[i] != ':'; i++)
      ;

    std::string strip_optname(optname, i);
    it = _options_name.find(strip_optname);

    if (it == _options_name.end())
    {
      // couldn't find option
      throw BadOption(optname, "");
    }
    else
    {
      IOption *opt = it->second;

      if (opt->getSeparators().empty())
      {
        // couldn't find option
        throw BadOption(optname, "");
      }
    }
  }

  if (it->second->isDisabled())
  {
    // if option is disabled we don't have to recognize it
    throw BadOption(optname, "");
  }

  return it->second;

} // findOption

// check that option value is correct
static void checkOptionValue(const IOption *opt, const std::string &opt_name,
                             const std::string &val)
{
  auto valid_vals = opt->getValidVals();
  bool is_valid = valid_vals.empty();

  for (const auto &v : valid_vals)
  {
    if (v == val)
    {
      // value is valid
      is_valid = true;
      break;
    }
  }

  if (!is_valid)
  {
    throw BadOption(opt_name, val);
  }

} // checkOptionValue

const char *CommandLine::findOptionValue(const IOption *opt, const char **argv, int cur_argv)
{
  auto seps = opt->getSeparators();
  const char *opt_name = argv[cur_argv];
  const char *val_pos = nullptr;

  // search one of the separators
  for (auto s : seps)
  {
    for (int i = 0; opt_name[i] != '\0'; i++)
    {
      if (s == opt_name[i])
      {
        // separator is found, set val_pos to symbol after it
        val_pos = &opt_name[i] + 1;
        break;
      }
    }

    if (val_pos)
    {
      break;
    }
  }

  // if option doesn't have additional separators or these separators aren't
  // found then we assume that option value is the next element in argv,
  // but if the next element starts with '-' we suppose that option value is empty
  // because options start with '-'
  if (!val_pos)
  {
    if (_args_num == cur_argv + 1)
    {
      val_pos = "";
    }
    else
    {
      val_pos = argv[cur_argv + 1];

      if (val_pos[0] == '-')
      {
        // it can be a value for numeric (negative numbers)
        // or symbolic (contains value `-`) option
        if (!isdigit(val_pos[1]) && val_pos[1])
        {
          val_pos = "";
        }
      }
    }
  }

  // check that option value is correct
  checkOptionValue(opt, opt_name, val_pos);

  return val_pos;

} // findOptionValue

const char *CommandLine::findValueForMultOption(const IOption *opt, const std::string &opt_name,
                                                const char **argv, int cur_argv)
{
  const char *val_pos = nullptr;

  if (cur_argv >= _args_num)
  {
    return nullptr;
  }

  val_pos = argv[cur_argv];

  if (val_pos[0] == '-')
  {
    // it can be a value for numeric (negative numbers)
    // or symbolic (contains value `-`) option
    if (!isdigit(val_pos[1]) && val_pos[1])
    {
      return nullptr;
    }
  }

  checkOptionValue(opt, opt_name, val_pos);

  return val_pos;

} // findValueForMultOption

/**
 * @brief find option by name
 * @param opt - found option
 * @param options - all options
 * @return true if option was found in options
 */
static bool isOptionInOptions(IOption *opt, const std::set<std::string> &options)
{

  for (const auto &name : opt->getNames())
  {
    if (options.find(name) != options.end())
    {
      return true;
    }
  }

  return false;

} // isOptionInOptions

static bool areOptionsIntersected(const std::vector<IOption *> grouped_options,
                                  const std::set<std::string> &all_options)
{
  for (const auto &opt : grouped_options)
    if (isOptionInOptions(opt, all_options))
      return true;

  return false;
} // areOptionsIntersected

void CommandLine::checkRegisteredOptions(const std::set<std::string> &cmd_args)
{
  for (const auto &opt : _options)
  {
    if (opt->isOptional() || isOptionInOptions(opt, cmd_args))
      continue;

    if (opt->isGrouped())
    {
      auto it = _grouped_options.find(opt->getGroup());
      assert(it != _grouped_options.end());

      if (!areOptionsIntersected(it->second, cmd_args))
        continue;
    }

    // option is not found then print error message
    std::string options;

    for (const auto &n : opt->getNames())
    {
      options += (n + " ");
    }

    usage("one of the following options must be defined: " + options);
  }

} // checkRegisteredOptions

void CommandLine::checkOptions(const std::set<std::string> &cmd_args)
{
  for (const auto &o : _options)
  {
    // search option from command line
    for (const auto &n : o->getNames())
    {
      if (cmd_args.find(n) == cmd_args.end())
      {
        // name isn't found
        continue;
      }

      // check option
      try
      {
        o->runCheckerFunc();
      }
      catch (BadOption &e)
      {
        usage(e.what());
      }

    } // opt names
  }   // options

} // checkOptions

void CommandLine::parseCommandLine(int argc, const char **argv, bool check_nonoptional)
{
  std::set<std::string> cmd_args;
  IOption *opt;
  const char *arg_val = nullptr;

  _prog_name = argv[0];
  _args_num = argc;

  if (argc == 1)
  {
    // empty command line
    usage();
  }

  // search help option and print help if this option is passed
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
    {
      usage("", EXIT_SUCCESS);
    }
  }

  for (int i = 1; i < argc; i += (argv[i + 1] == arg_val) ? 2 : 1)
  {
    if (argv[i][0] != '-')
    {
      std::string err_msg(std::string("invalid command line argument: ") + argv[i]);
      usage(err_msg);
    }

    // find registered option
    try
    {
      opt = findOption(argv[i]);
    }
    catch (BadOption &e)
    {
      std::string err_msg(std::string("invalid option: ") + e.getName());
      usage(err_msg);
    }

    // figure out value for option
    try
    {
      if (opt->canHaveSeveralVals())
      {
        int j = i + 1;
        for (arg_val = findValueForMultOption(opt, argv[i], argv, j); arg_val;
             arg_val = findValueForMultOption(opt, argv[i], argv, j))
        {
          // set value for option
          opt->setValue(arg_val);
          j++;
        }

        i = j - 1;
      }
      else
      {
        arg_val = findOptionValue(opt, argv, i);

        // set value for option
        opt->setValue(arg_val);
      }
    }
    catch (BadOption &e)
    {
      std::string optname = e.getName();
      optname = optname.empty() ? argv[i] : optname;
      std::string err_msg(std::string("invalid value: ") + e.getValue() +
                          std::string(" for option: ") + optname);
      usage(err_msg);
    }

    // we can't just put argv[i] because option can have separators
    cmd_args.insert(opt->getNames()[0]);
  }

  if (check_nonoptional)
  {
    // check that all registered options are present in command line
    checkRegisteredOptions(cmd_args);
  }

  // verify options
  checkOptions(cmd_args);

} // parseCommandLine

//
// specializations of setValue method for all supported option type
//
// string
template <> void Option<std::string>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(val);
}

// vector of strings
template <> void Option<std::vector<std::string>>::setValue(const std::string &val)
{
  if (!val.empty())
    this->push_back(val);
}

// vector of ints
template <> void Option<std::vector<int>>::setValue(const std::string &val)
{
  if (!val.empty())
    this->push_back(stoi(val));
}

// bool
template <> void Option<bool>::setValue(const std::string &val)
{
  this->setRawValue(this->convToBool(val));
}

// char
template <> void Option<char>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->convToChar(val));
}

// int8
template <> void Option<int8_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<int64_t>(val));
}

// int16
template <> void Option<int16_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<int64_t>(val));
}

// int32
template <> void Option<int32_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<int64_t>(val));
}

// uint8
template <> void Option<uint8_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<uint64_t>(val));
}

// uint16
template <> void Option<uint16_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<uint64_t>(val));
}

// uint32
template <> void Option<uint32_t>::setValue(const std::string &val)
{
  if (!val.empty())
    this->setRawValue(this->template convToNum<uint64_t>(val));
}

} // namespace cli
} // namespace nnc
