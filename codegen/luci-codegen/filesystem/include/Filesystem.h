/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_CODEGEN_FILESYSTEM_H
#define LUCI_CODEGEN_FILESYSTEM_H

#include <string>
#include <iomanip>
#include <istream>
#include <ostream>
#include <system_error>

namespace luci_codegen_filesystem
{

class path
{
public:
  typedef char value_type;
  typedef std::basic_string<value_type> string_type;

  enum format
  {
    // there are two more formats in original std::filesystem::path::format, omit it sice this is just a minimal subset of functionality
    auto_format
  };

  path() noexcept = default;

  path(const path &p) : _path(p._path)
  {
    // DO NOTHING
  }

  path(path &&p) noexcept : _path(std::move(p._path))
  {
    // DO NOTHING
  }

  path(string_type &&source, format fmt = auto_format) : _path(std::move(source))
  {
    // DO NOTHING
  }

  template <class Source> path(const Source &source, format fmt = auto_format) : _path(source)
  {
    // DO NOTHING
  }

  // Note that there are three more constructors in original std;:filesystem::path

  path &operator=(const path &p)
  {
    _path = p._path;
    return *this;
  }

  path &operator=(path &&p) noexcept
  {
    _path = std::move(p._path);
    return *this;
  }

  path &operator=(string_type &&source)
  {
    _path = std::move(source);
    return *this;
  }

  template <class Source> path &operator=(const Source &source)
  {
    _path = source;
    return *this;
  }

  template< class Source >
  path& append(const Source &source)
  {
    if (source[0] == '/')
    {
      _path = source;
    }
    else
    {
      _path += "/";
      _path += source;
    }
    return *this;
  }

  path& operator/=(const path& p)
  {
    return append(p._path);
  }

  template< class Source >
  path& operator/=(const Source& source)
  {
    return append(source);
  }

  void clear() noexcept
  {
    _path.clear();
  }

  bool empty() const noexcept
  {
    return _path.empty();
  }

  const value_type* c_str() const noexcept
  {
    return _path.c_str();
  }

  const string_type& native() const noexcept
  {
    return _path;
  }

  std::string string() const
  {
    return _path;
  }

  operator string_type() const
  {
    return _path;
  }

private:
  std::string _path;
};

path operator/(const path &lhs, const path &rhs);

template< class CharT, class Traits >
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os, const path &p)
{
  os << std::quoted(p.string());
  return os;
}

template< class CharT, class Traits >
std::basic_istream<CharT,Traits> &
operator>>(std::basic_istream<CharT,Traits> &is, path &p )
{
  is >> std::quoted(p.string());
  return is;
}

class filesystem_error: public std::system_error
{
public:
  filesystem_error(const std::string &what_arg,
                   std::error_code ec): std::system_error(ec, what_arg)
  {
    // DO NOTHING
  }

  filesystem_error( const std::string &what_arg,
                    const path &p1,
                    std::error_code ec ) : std::system_error(ec, what_arg), _p1(p1)
  {
    // DO NOTHING
  }

  filesystem_error(const std::string &what_arg,
                    const path &p1,
                    const path &p2,
                    std::error_code ec) :std::system_error(ec, what_arg), _p1(p1), _p2(p2)
  {
    // DO NOTHING
  }

  filesystem_error(const filesystem_error &other) noexcept: std::system_error(other), _p1(other._p1), _p2(other._p2)
  {
    // DO NOTHING
  }

  const path &path1() const noexcept
  {
    return _p1;
  }

  const path &path2() const noexcept
  {
    return _p2;
  }

private:
  path _p1;
  path _p2;
};

bool exists(const path &p);

bool is_directory(const path &p);

bool create_directory(const path &p);

} // namespace luci_codegen_filesystem

#endif // LUCI_CODEGEN_FILESYSTEM_H
