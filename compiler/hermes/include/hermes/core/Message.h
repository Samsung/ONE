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

#ifndef __HERMES_MESSAGE_H__
#define __HERMES_MESSAGE_H__

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace hermes
{

/**
 * @brief Mutie-line text message
 */
class MessageText
{
public:
  /// WARNING! Be careful. This constructor updates "ss".
  MessageText(std::stringstream &ss);

public:
  /// @brief The number of lines
  uint32_t lines(void) const { return _lines.size(); }
  /// @brief The content of a specific line
  const std::string &line(uint32_t n) const { return _lines.at(n); }

private:
  std::vector<std::string> _lines;
};

/**
 * @brief Message with metadata
 *
 * TODO Add "Timestamp" field
 * TODO Add "Severity" field
 * TODO Support extensible "attribute" annotation
 */
class Message final
{
public:
  Message() = default;

public:
  void text(std::unique_ptr<MessageText> &&text) { _text = std::move(text); }
  const MessageText *text(void) const { return _text.get(); }

private:
  std::unique_ptr<MessageText> _text;
};

} // namespace hermes

#endif // __HERMES_MESSAGE_H__
