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

#ifndef NNCC_COMMANDLINE_H
#define NNCC_COMMANDLINE_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <type_traits>
#include <cassert>
#include <limits>
#include <iostream>

namespace nnc
{
namespace cli
{

/**
 * @brief simple exception class for invalid options
 */
class BadOption : public std::logic_error
{
public:
  explicit BadOption(const std::string &msg, std::string optname = "", std::string value = "")
    : std::logic_error(msg), _option_name(std::move(optname)), _option_value(std::move(value))
  {
  }

  /**
   * @brief get name for invalid option
   */
  const std::string &getName() const { return _option_name; }

  /**
   * @brief get value for invalid option
   */
  const std::string &getValue() const { return _option_value; }

private:
  std::string _option_name;
  std::string _option_value;
};

/**
 * @brief a class models option type
 */
template <typename T, bool isClass> class OptionType
{
public:
  OptionType() = default;
};

// for class type
template <typename T> class OptionType<T, true> : public T
{
public:
  /**
   * @brief set value for option
   * @tparam Tval - type of value what we want to assign to value
   * @param val - option value
   */
  template <typename Tval> void setRawValue(const Tval &val) { this->T::operator=(val); }

  /**
   * @brief get option value
   * @return value of option
   */
  const T &getRawValue() const { return *this; }

  T getRawValue() { return *this; }
};

// for scalar type
template <typename T> class OptionType<T, false>
{
public:
  /**
   * @brief convert Option to scalar option type
   */
  /*implicit*/ operator T() const
  {
    return _value;
  } // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

  /**
   * @brief set value for option
   * @tparam Tval - type of value what we want to assign to value
   * @param val - option value
   */
  template <typename Tval> void setRawValue(const Tval &val) { _value = val; }

  /**
   * @brief get option value
   * @return value of option
   */
  const T &getRawValue() const { return _value; }

  T getRawValue() { return _value; }

protected:
  // methods for Option
  bool convToBool(const std::string &val);
  char convToChar(const std::string &val);
  template <typename Tnum> Tnum convToNum(const std::string &val);

  // data
  T _value; // option value
};

/**
 * @brief interface for Option class
 */
class IOption
{
public:
  /**
   * @brief set option value
   * @param val - value of option in string format
   * @todo add support for vector
   */
  virtual void setValue(const std::string &val) = 0;

  /**
   * @brief get all names of option
   */
  virtual const std::vector<std::string> &getNames() const = 0;

  /**
   * @brief get description of option
   */
  virtual const std::string &getOverview() const = 0;

  /**
   * @brief may option be optional?
   */
  virtual bool isOptional() const = 0;

  /**
   * @brief get valid values for given option
   */
  virtual const std::vector<std::string> &getValidVals() const = 0;

  /**
   * @brief get separators for option
   */
  virtual const std::vector<char> &getSeparators() const = 0;

  /**
   * @brief function for option verification
   * @throw this function throws exception of BadOption
   *        type if verification is not passed
   */
  virtual void runCheckerFunc() = 0;

  /**
   * @brief is option disabled?
   */
  virtual bool isDisabled() const = 0;

  /**
   * @brief can option have several values?
   */
  virtual bool canHaveSeveralVals() const = 0;

  /**
   * @result true if option is in group
   */
  virtual bool isGrouped() const = 0;

  // groups for option. Each option can be put in one of these groups
  enum class Group
  {
    none = 0,
    caffe2 = 1,
    onnx = 2 // 'onnx' is currently unused
  };

  /**
   * @return group in which option is put
   */
  virtual IOption::Group getGroup() const = 0;

  /**
   * @brief name of option group
   */
  virtual std::string getGroupName() const = 0;

protected:
  // this array contains name of option groups. It must be synchronized with Group enum
  constexpr static const char *const _groupNames[] = {nullptr, "caffe2", "onnx"};
};

/**
 * @brief this class describes command line option
 * @tparam T - type of option
 */
template <typename T>
class Option final : public OptionType<T, std::is_class<T>::value>, public IOption
{
public:
  /**
   * @brief function type for option verification
   */
  using option_checker_t = void (*)(const Option<T> &);

  /**
   * @brief construct an option
   * @tparam T  - type of an option
   * @param optnames - names of option
   * @param descr - overview of option
   * @param default_val - option value accepted by default
   * @param is_optional - is option optional?
   * @param vals - valid values for option. Other values are interpreted as invalid
   * @param checker - function verifies option
   * @param seps - symbols that separates name option from value (by default is spaces)
   * @param enabled - if this option is set to false then it won't be shown for users
   * @param group - all options can be splitted into groups so this param sets group for option
   */
  explicit Option(const std::vector<std::string> &optnames, const std::string &descr,
                  const T &default_val = T(), bool is_optional = false,
                  const std::vector<std::string> &vals = std::vector<std::string>(),
                  option_checker_t checker = nullptr,
                  const std::vector<char> &seps = std::vector<char>(), bool enabled = true,
                  IOption::Group group = IOption::Group::none);

  // options must not be copyable and assignment
  Option(const Option &) = delete;

  Option &operator=(const Option &) = delete;

  /**
   * @brief overload assignment operator for type
   */
  template <typename Tval> T &operator=(const Tval &val)
  { // NOLINT(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    setRawValue(val);
    return this->getRawValue(); // If not using `this` it won't work
  }

  // overridden methods
  void setValue(const std::string &val) override;

  const std::vector<std::string> &getNames() const override { return _names; }

  const std::string &getOverview() const override { return _descr; }

  bool isOptional() const override { return _is_optional; }

  const std::vector<std::string> &getValidVals() const override { return _valid_vals; }

  void runCheckerFunc() override
  {
    if (_checker)
    {
      _checker(*this);
    }
  }

  const std::vector<char> &getSeparators() const override { return _seps; }

  bool isDisabled() const override { return !_is_enabled; }

  bool canHaveSeveralVals() const override { return _can_have_several_vals; }

  bool isGrouped() const override { return _group != IOption::Group::none; }

  IOption::Group getGroup() const override { return _group; }

  std::string getGroupName() const override { return _groupNames[static_cast<size_t>(_group)]; }
  // end overridden methods

private:
  // data
  std::vector<std::string> _names; // names of the option
  std::string _descr;              // overview of option
  bool _is_optional;
  std::vector<std::string> _valid_vals; // option can be initialized only by these values
  option_checker_t _checker;            // function verifies option and its value
  std::vector<char> _seps;              // these symbols separate option name and its value
  bool _is_enabled;
  bool _can_have_several_vals; // can option take several values?
  IOption::Group _group;       // group for option
};

/**
 * @brief this class describes a common command line interface
 */
class CommandLine
{ // NOLINT(cppcoreguidelines-special-member-functions, hicpp-special-member-functions)
public:
  // prevent copy or assignment
  CommandLine(const CommandLine &) = delete;

  CommandLine &operator=(const CommandLine &) = delete;

  /**
   * @brief singleton method
   */
  static CommandLine *getParser();

  /**
   * @brief parse command line option
   * @param argc - number of command line arguments
   * @param argv - command line arguments
   * @param check_nonoptional - if true then check that all non optional declared options are
   * presented
   */
  void parseCommandLine(int argc, const char **argv, bool check_nonoptional = true);

  /**
   * @brief register option for parser
   * @param opt - option
   */
  void registerOption(IOption *opt);

private:
  /**
   * @brief print usage and exit
   * @param msg - additional user message
   * @param exit_code - the program is terminated with this code
   */
  [[noreturn]] void usage(const std::string &msg = "", int exit_code = EXIT_FAILURE);

  /**
   * @brief check that all non optional registered options are passed from command line
   * @param cmd_args - arguments from command line
   */
  void checkRegisteredOptions(const std::set<std::string> &cmd_args);

  /**
   * @brief call verification function, if present, for option
   * @param cmd_args - arguments from command line
   */
  void checkOptions(const std::set<std::string> &cmd_args);

  /**
   * @brief find option with `optname` and set `pos` to option value
   * @param optname - name of option
   * @return pointer to option
   * @throw BadOption throw exception if option not found
   */
  IOption *findOption(const char *optname);

  /**
   * @brief figure out option value
   * @param opt - option for which value is looked for
   * @param argv - array of command line arguments
   * @param cur_argv - current position in argv (i.e. cur_argv point to option name)
   * @return position in argv where option value begins or empty string if option doesn't have value
   * @throw BadOption throw exception if value for option is incorrect
   */
  const char *findOptionValue(const IOption *opt, const char **argv, int cur_argv);

  /**
   * @brief figure out value for option with multiple values
   * @param opt - option for which value is looked for
   * @param opt_name - option name which taken from command line
   * @param argv - array of command line arguments
   * @param val_argv - position in argv for current option value
   * @return position in argv where option value begins or nullptr if option doesn't have value
   * anymore
   * @throw BadOption throw exception if value for option is incorrect
   */
  const char *findValueForMultOption(const IOption *opt, const std::string &opt_name,
                                     const char **argv, int cur_argv);

  // allow object constructor only for methods
  CommandLine() = default;

  // data
  std::map<std::string, IOption *> _options_name; // map of name -> option
  std::vector<IOption *> _options;                // options
  std::map<IOption::Group, std::vector<IOption *>>
    _grouped_options;     // map of groups: group -> vector of options
  std::string _prog_name; // name of program
  int _args_num = 0;      // number of command line arguments
};

// the following functions are helpers for users that declare new options
/**
 * @brief convert option names for Option constructor
 * @param names - name of option, if option has several names then
 *               `names` must be represented by a string separated by a comma
 */
std::vector<std::string> optname(const char *names);

/** @brief convert option overview for Option constructor */
inline std::string overview(const char *descr)
{
  std::string overview(descr);
  assert(!overview.empty());

  return overview;
}

/** @brief convert option overview for Option constructor */
inline bool optional(bool is_optional) { return is_optional; }

/**
 * @brief register valid values for option
 * @param vals - valid values of option, if option has several that values then
 *               `vals` must be represented by a string separated by a comma
 */
std::vector<std::string> optvalues(const char *vals);

/**
 * @brief separators that separate option name and its value
 * @param seps - chars of separators separated by a comma
 */
std::vector<char> separators(const char *seps);

/**
 * @param is_shown - if set to false, then option won't be shown in help message
 */
inline bool showopt(bool is_shown) { return is_shown; }
// end of helper functions

//
// Implementation of template functions
//
template <typename T> bool OptionType<T, false>::convToBool(const std::string &val)
{
  if (val.empty() || val == "TRUE" || val == "True" || val == "true" || val == "1")
  {
    return true;
  }

  if (val == "FALSE" || val == "False" || val == "false" || val == "0")
  {
    return false;
  }

  throw BadOption("", val);

} // convToBool

template <typename T> char OptionType<T, false>::convToChar(const std::string &val)
{
  if (val.length() == 1)
  {
    return val[0];
  }
  else
  {
    throw BadOption("", val);
  }

} // convToChar

template <typename T>
template <typename Tnum>
Tnum OptionType<T, false>::convToNum(const std::string &val)
{
  Tnum num_val;

  assert((std::is_same<Tnum, uint64_t>::value || std::is_same<Tnum, int64_t>::value));
  assert((std::numeric_limits<T>::max() < std::numeric_limits<Tnum>::max()));
  assert(std::numeric_limits<T>::min() >= std::numeric_limits<Tnum>::min());

  try
  {
    num_val = std::is_same<Tnum, uint64_t>::value ? stoull(val) : stoll(val);
  }
  catch (...)
  {
    throw BadOption("", val);
  }

  if (num_val > std::numeric_limits<T>::max() || num_val < std::numeric_limits<T>::min())
  {
    throw BadOption("", val);
  }

  return num_val;

} // convToNum

template <typename T>
Option<T>::Option(const std::vector<std::string> &optnames, const std::string &descr,
                  const T &default_val, bool is_optional, const std::vector<std::string> &vals,
                  option_checker_t checker, const std::vector<char> &seps, bool enabled,
                  IOption::Group group)
{
  // save all names
  for (const auto &n : optnames)
  {
    _names.push_back(n);

    assert(n[0] == '-' && "option name must start with `-`");
  }

  _descr = descr;
  _is_optional = is_optional;
  _valid_vals = vals;
  _seps = seps;

  this->setRawValue(default_val);

#ifndef NDEBUG
  // check that separators are valid symbols
  for (const auto &s : _seps)
  {
    assert((s == '=' || s == ':') && "invalid option separators");
  }
#endif // NDEBUG

  // save checker
  _checker = checker;

  _is_enabled = enabled;
  assert((_is_enabled || _is_optional || group != IOption::Group::none) &&
         "disabled non-group option can't be required");

  _group = group;

  _can_have_several_vals =
    std::is_same<T, std::vector<std::string>>::value || std::is_same<T, std::vector<int>>::value;
  assert(!(_can_have_several_vals && !_seps.empty()) &&
         "option with several values can't have separators");

  // register new option for parser
  CommandLine::getParser()->registerOption(this);

} // Option

//
// prototypes of option checker functions
//
void checkInFile(const Option<std::string> &in_file);

void checkOutFile(const Option<std::string> &out_file);

void checkInDir(const Option<std::string> &dir);

void checkOutDir(const Option<std::string> &dir);

} // namespace cli
} // namespace nnc

#endif // NNCC_COMMANDLINE_H
