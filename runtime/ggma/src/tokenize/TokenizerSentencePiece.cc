/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Tokenizer.h"
#include "TokenizerFactory.h"

#include <filesystem>
#include <sentencepiece_processor.h>

namespace ggma
{

class SentencePieceTokenizer : ggma::Tokenizer
{
public:
  std::string id() const { return "sentencepiece"; }
  static Tokenizer *create(const std::string &tokenizer_dir);
  size_t tokenize(const char *text, size_t text_len, int32_t *tokens, size_t max_tokens,
                  size_t *n_tokens) const override;
  size_t detokenize(const int32_t *tokens, size_t n_tokens, char *text,
                    size_t text_len) const override;

private:
  std::unique_ptr<::sentencepiece::SentencePieceProcessor> _processor;
};

ggma::Tokenizer *SentencePieceTokenizer::create(const std::string &tokenizer_dir)
{
  auto tokenizer = std::make_unique<SentencePieceTokenizer>();
  tokenizer->_processor = std::make_unique<::sentencepiece::SentencePieceProcessor>();

  std::filesystem::path model_path = std::filesystem::path(tokenizer_dir) / "tokenizer.model";
  auto status = tokenizer->_processor->Load(model_path.string());

  return status.ok() ? tokenizer.release() : nullptr;
}

size_t SentencePieceTokenizer::tokenize(const char *text, size_t text_len, int32_t *tokens,
                                        size_t max_tokens, size_t *n_tokens) const
{
  if (!text || !tokens || !n_tokens || max_tokens == 0 || !_processor)
  {
    if (n_tokens)
      *n_tokens = 0;
    return 0;
  }

  std::string input_text(text, text_len);
  std::vector<int> piece_ids;

  auto status = _processor->Encode(input_text, &piece_ids);
  if (!status.ok())
  {
    if (n_tokens)
      *n_tokens = 0;
    return 0;
  }

  // Initialize tokens array to 0
  std::fill(tokens, tokens + max_tokens, 0);

  // Check BOS token ID and add it if it's 1
  int bos_id = _processor->bos_id();
  size_t bos_offset = 0;

  // TODO: Make BOS token prepending configurable
  if (bos_id >= 0 && max_tokens > 0)
  {
    tokens[0] = bos_id; // Add BOS token
    bos_offset = 1;     // Start actual tokens from index 1
  }

  size_t available_space = max_tokens - bos_offset;
  size_t token_count = std::min(piece_ids.size(), available_space);

  for (size_t i = 0; i < token_count; ++i)
    tokens[bos_offset + i] = static_cast<int32_t>(piece_ids[i]);

  *n_tokens = bos_offset + token_count;
  return bos_offset + token_count;
}

size_t SentencePieceTokenizer::detokenize(const int32_t *tokens, size_t n_tokens, char *text,
                                          size_t text_len) const
{
  if (!tokens || !text || n_tokens == 0 || text_len == 0 || !_processor)
    return 0;

  std::vector<int> piece_ids;
  for (size_t i = 0; i < n_tokens; ++i)
    piece_ids.push_back(tokens[i]);

  std::string decoded_text;
  auto status = _processor->Decode(piece_ids, &decoded_text);
  if (!status.ok())
    return 0;

  size_t copy_len = std::min(decoded_text.length(), text_len - 1);
  memcpy(text, decoded_text.c_str(), copy_len);
  text[copy_len] = '\0';

  return copy_len;
}

} // namespace ggma

// Register SentencePieceTokenizer using the macro
REGISTER_TOKENIZER("sentencepiece", SentencePieceTokenizer)
