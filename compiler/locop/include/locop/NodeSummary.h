/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_NODE_SUMMARY_H__
#define __LOCO_NODE_SUMMARY_H__

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace locop
{

using OpName = std::string;
using ArgName = std::string;
using ArgValue = std::string;
using ArgElem = std::pair<ArgName, ArgValue>;

class ArgDesc
{
public:
  ArgDesc() = default;

public:
  /// @brief The number of presented arguments
  uint32_t count(void) const { return _args.size(); }

  const ArgElem &at(uint32_t n) const { return _args.at(n); }
  void append(const ArgName &name, const ArgValue &value) { _args.emplace_back(name, value); }

private:
  std::vector<ArgElem> _args;
};

struct NodeDesc
{
public:
  /**
   * @brief Multi-line comments
   */
  class Comments final
  {
  public:
    Comments() = default;

  public:
    uint32_t count(void) const { return _lines.size(); }
    const std::string &at(uint32_t n) const { return _lines.at(n); }
    void append(const std::string &s);

  private:
    std::vector<std::string> _lines;
  };

public:
  enum class State
  {
    // All the node descriptions are "Invalid" at the beginning.
    //
    // Any valid node description SHOULD NOT be at this state.
    Invalid,
    // This state means that the producer is **NOT** confident about the information that
    // it generates.
    //
    // There may be some missing information.
    PartiallyKnown,
    // This state means that the producer is confident about the information that it
    // generates.
    Complete,
  };

public:
  NodeDesc() = default;
  NodeDesc(const OpName &opname) { this->opname(opname); }

public:
  const OpName &opname(void) const;
  void opname(const OpName &value);

  const ArgDesc &args(void) const { return _args; }
  ArgDesc &args(void) { return _args; }

  const Comments &comments(void) const { return _comments; }
  Comments &comments(void) { return _comments; }

  const State &state(void) const { return _state; }
  void state(const State &s) { _state = s; }

private:
  std::unique_ptr<OpName> _name = nullptr;
  ArgDesc _args;
  Comments _comments;
  State _state = State::Invalid;
};

using NodeSummary = NodeDesc;

} // namespace locop

#endif // __LOCO_NODE_SUMMARY_H__
