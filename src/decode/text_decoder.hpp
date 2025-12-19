/**
 * @file text_decoder.hpp
 * @brief Text/NLP data loading and tokenization (v2.23.0)
 *
 * Provides utilities for loading and processing text data for NLP models:
 * - Basic tokenization (whitespace, word-piece inspired)
 * - Vocabulary management
 * - Sequence padding and truncation
 * - Sequence packing for efficiency
 *
 * Usage:
 * ```cpp
 * // Build vocabulary from corpus
 * Vocabulary vocab;
 * vocab.build_from_file("corpus.txt", max_vocab_size);
 *
 * // Tokenize text
 * BasicTokenizer tokenizer(vocab);
 * auto tokens = tokenizer.encode("Hello world!");
 *
 * // Decode back to text
 * auto text = tokenizer.decode(tokens);
 *
 * // Batch processing with padding
 * SequenceBatcher batcher(max_length, pad_token_id);
 * auto batch = batcher.create_batch(sequences);
 * ```
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace turboloader {

/**
 * @brief Special tokens used in tokenization
 */
struct SpecialTokens {
    std::string pad_token = "[PAD]";
    std::string unk_token = "[UNK]";
    std::string bos_token = "[BOS]";
    std::string eos_token = "[EOS]";
    std::string sep_token = "[SEP]";
    std::string cls_token = "[CLS]";
    std::string mask_token = "[MASK]";
};

/**
 * @brief Token-to-ID and ID-to-Token vocabulary
 */
class Vocabulary {
public:
    Vocabulary() {
        // Add special tokens by default
        add_special_tokens();
    }

    explicit Vocabulary(const SpecialTokens& special)
        : special_(special) {
        add_special_tokens();
    }

    /**
     * @brief Build vocabulary from text file
     * @param path Path to text file
     * @param max_size Maximum vocabulary size (0 for unlimited)
     * @param min_freq Minimum frequency to include token
     */
    void build_from_file(const std::string& path, size_t max_size = 0,
                          size_t min_freq = 1) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        // Count token frequencies
        std::unordered_map<std::string, size_t> freq;
        std::string line;
        while (std::getline(file, line)) {
            auto tokens = tokenize_line(line);
            for (const auto& tok : tokens) {
                freq[tok]++;
            }
        }

        // Sort by frequency
        std::vector<std::pair<std::string, size_t>> sorted_tokens(freq.begin(), freq.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Add tokens to vocabulary
        for (const auto& [tok, count] : sorted_tokens) {
            if (count < min_freq) break;
            if (max_size > 0 && size() >= max_size) break;
            if (!contains(tok)) {
                add_token(tok);
            }
        }
    }

    /**
     * @brief Build vocabulary from vector of strings
     */
    void build_from_texts(const std::vector<std::string>& texts,
                           size_t max_size = 0, size_t min_freq = 1) {
        std::unordered_map<std::string, size_t> freq;

        for (const auto& text : texts) {
            auto tokens = tokenize_line(text);
            for (const auto& tok : tokens) {
                freq[tok]++;
            }
        }

        std::vector<std::pair<std::string, size_t>> sorted_tokens(freq.begin(), freq.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [tok, count] : sorted_tokens) {
            if (count < min_freq) break;
            if (max_size > 0 && size() >= max_size) break;
            if (!contains(tok)) {
                add_token(tok);
            }
        }
    }

    /**
     * @brief Add a token to vocabulary
     */
    int32_t add_token(const std::string& token) {
        if (token_to_id_.count(token)) {
            return token_to_id_[token];
        }
        int32_t id = static_cast<int32_t>(id_to_token_.size());
        token_to_id_[token] = id;
        id_to_token_.push_back(token);
        return id;
    }

    /**
     * @brief Get token ID (returns UNK if not found)
     */
    int32_t get_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        return unk_id();
    }

    /**
     * @brief Get token string (returns UNK token if invalid ID)
     */
    std::string get_token(int32_t id) const {
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
            return id_to_token_[id];
        }
        return special_.unk_token;
    }

    /**
     * @brief Check if token exists in vocabulary
     */
    bool contains(const std::string& token) const {
        return token_to_id_.count(token) > 0;
    }

    // Special token IDs
    int32_t pad_id() const { return get_id(special_.pad_token); }
    int32_t unk_id() const { return get_id(special_.unk_token); }
    int32_t bos_id() const { return get_id(special_.bos_token); }
    int32_t eos_id() const { return get_id(special_.eos_token); }
    int32_t sep_id() const { return get_id(special_.sep_token); }
    int32_t cls_id() const { return get_id(special_.cls_token); }
    int32_t mask_id() const { return get_id(special_.mask_token); }

    size_t size() const { return id_to_token_.size(); }

    /**
     * @brief Save vocabulary to file
     */
    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create file: " + path);
        }
        for (const auto& token : id_to_token_) {
            file << token << "\n";
        }
    }

    /**
     * @brief Load vocabulary from file
     */
    void load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        token_to_id_.clear();
        id_to_token_.clear();

        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                add_token(line);
            }
        }
    }

    const SpecialTokens& special_tokens() const { return special_; }

private:
    void add_special_tokens() {
        add_token(special_.pad_token);
        add_token(special_.unk_token);
        add_token(special_.bos_token);
        add_token(special_.eos_token);
        add_token(special_.sep_token);
        add_token(special_.cls_token);
        add_token(special_.mask_token);
    }

    std::vector<std::string> tokenize_line(const std::string& line) const {
        std::vector<std::string> tokens;
        std::string current;

        for (char c : line) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
            } else if (std::ispunct(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                current += std::tolower(static_cast<unsigned char>(c));
            }
        }

        if (!current.empty()) {
            tokens.push_back(current);
        }

        return tokens;
    }

    SpecialTokens special_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
};

/**
 * @brief Basic tokenizer with vocabulary lookup
 */
class BasicTokenizer {
public:
    explicit BasicTokenizer(const Vocabulary& vocab)
        : vocab_(&vocab) {}

    /**
     * @brief Tokenize and encode text to IDs
     */
    std::vector<int32_t> encode(const std::string& text,
                                  bool add_bos = false,
                                  bool add_eos = false) const {
        std::vector<int32_t> ids;

        if (add_bos) {
            ids.push_back(vocab_->bos_id());
        }

        // Simple whitespace + punctuation tokenization
        std::string current;
        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    ids.push_back(vocab_->get_id(current));
                    current.clear();
                }
            } else if (std::ispunct(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    ids.push_back(vocab_->get_id(current));
                    current.clear();
                }
                ids.push_back(vocab_->get_id(std::string(1, c)));
            } else {
                current += std::tolower(static_cast<unsigned char>(c));
            }
        }

        if (!current.empty()) {
            ids.push_back(vocab_->get_id(current));
        }

        if (add_eos) {
            ids.push_back(vocab_->eos_id());
        }

        return ids;
    }

    /**
     * @brief Decode IDs back to text
     */
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special = true) const {
        std::string result;
        bool first = true;

        for (int32_t id : ids) {
            std::string token = vocab_->get_token(id);

            // Skip special tokens if requested
            if (skip_special) {
                if (token == vocab_->special_tokens().pad_token ||
                    token == vocab_->special_tokens().bos_token ||
                    token == vocab_->special_tokens().eos_token ||
                    token == vocab_->special_tokens().cls_token ||
                    token == vocab_->special_tokens().sep_token) {
                    continue;
                }
            }

            // Add space before words (but not punctuation)
            if (!first && token.size() == 1 && !std::ispunct(token[0])) {
                result += " ";
            } else if (!first && token.size() > 1) {
                result += " ";
            }

            result += token;
            first = false;
        }

        return result;
    }

    /**
     * @brief Get vocabulary reference
     */
    const Vocabulary& vocab() const { return *vocab_; }

private:
    const Vocabulary* vocab_;
};

/**
 * @brief Tokenized sequence with attention mask
 */
struct TokenizedSequence {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> token_type_ids;  // For BERT-style segment embeddings

    size_t length() const { return input_ids.size(); }
};

/**
 * @brief Batch of tokenized sequences
 */
struct TokenizedBatch {
    std::vector<std::vector<int32_t>> input_ids;
    std::vector<std::vector<int32_t>> attention_mask;
    std::vector<std::vector<int32_t>> token_type_ids;

    size_t batch_size() const { return input_ids.size(); }
    size_t max_length() const {
        size_t max_len = 0;
        for (const auto& seq : input_ids) {
            max_len = std::max(max_len, seq.size());
        }
        return max_len;
    }
};

/**
 * @brief Sequence batcher with padding and truncation
 */
class SequenceBatcher {
public:
    enum class PaddingSide { LEFT, RIGHT };
    enum class TruncationSide { LEFT, RIGHT };

    SequenceBatcher(size_t max_length, int32_t pad_token_id,
                    PaddingSide padding = PaddingSide::RIGHT,
                    TruncationSide truncation = TruncationSide::RIGHT)
        : max_length_(max_length),
          pad_token_id_(pad_token_id),
          padding_side_(padding),
          truncation_side_(truncation) {}

    /**
     * @brief Pad/truncate a single sequence
     */
    TokenizedSequence process(const std::vector<int32_t>& input_ids) const {
        TokenizedSequence result;

        // Truncate if necessary
        if (input_ids.size() > max_length_) {
            if (truncation_side_ == TruncationSide::RIGHT) {
                result.input_ids = std::vector<int32_t>(
                    input_ids.begin(), input_ids.begin() + max_length_);
            } else {
                result.input_ids = std::vector<int32_t>(
                    input_ids.end() - max_length_, input_ids.end());
            }
        } else {
            result.input_ids = input_ids;
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        size_t real_length = result.input_ids.size();
        result.attention_mask.resize(max_length_, 0);

        // Pad if necessary
        if (result.input_ids.size() < max_length_) {
            size_t pad_length = max_length_ - result.input_ids.size();

            if (padding_side_ == PaddingSide::RIGHT) {
                result.input_ids.resize(max_length_, pad_token_id_);
                std::fill(result.attention_mask.begin(),
                          result.attention_mask.begin() + real_length, 1);
            } else {
                std::vector<int32_t> padded(pad_length, pad_token_id_);
                padded.insert(padded.end(), result.input_ids.begin(), result.input_ids.end());
                result.input_ids = std::move(padded);
                std::fill(result.attention_mask.begin() + pad_length,
                          result.attention_mask.end(), 1);
            }
        } else {
            std::fill(result.attention_mask.begin(), result.attention_mask.end(), 1);
        }

        // Initialize token_type_ids to 0
        result.token_type_ids.resize(max_length_, 0);

        return result;
    }

    /**
     * @brief Create a batch from multiple sequences
     */
    TokenizedBatch create_batch(const std::vector<std::vector<int32_t>>& sequences) const {
        TokenizedBatch batch;
        batch.input_ids.reserve(sequences.size());
        batch.attention_mask.reserve(sequences.size());
        batch.token_type_ids.reserve(sequences.size());

        for (const auto& seq : sequences) {
            auto processed = process(seq);
            batch.input_ids.push_back(std::move(processed.input_ids));
            batch.attention_mask.push_back(std::move(processed.attention_mask));
            batch.token_type_ids.push_back(std::move(processed.token_type_ids));
        }

        return batch;
    }

    size_t max_length() const { return max_length_; }
    int32_t pad_token_id() const { return pad_token_id_; }

private:
    size_t max_length_;
    int32_t pad_token_id_;
    PaddingSide padding_side_;
    TruncationSide truncation_side_;
};

/**
 * @brief Sequence packer for efficient training
 *
 * Packs multiple short sequences into a single sequence to minimize
 * padding overhead. Useful for pre-training language models.
 */
class SequencePacker {
public:
    explicit SequencePacker(size_t max_length, int32_t sep_token_id)
        : max_length_(max_length), sep_token_id_(sep_token_id) {}

    /**
     * @brief Pack sequences into fixed-length chunks
     * @param sequences Variable-length sequences to pack
     * @return Packed sequences with separator tokens
     */
    std::vector<std::vector<int32_t>> pack(
        const std::vector<std::vector<int32_t>>& sequences) const {

        std::vector<std::vector<int32_t>> packed;
        std::vector<int32_t> current;
        current.reserve(max_length_);

        for (const auto& seq : sequences) {
            // If adding this sequence would exceed max_length, start new chunk
            size_t needed = seq.size() + (current.empty() ? 0 : 1);  // +1 for separator
            if (current.size() + needed > max_length_) {
                if (!current.empty()) {
                    packed.push_back(std::move(current));
                    current = std::vector<int32_t>();
                    current.reserve(max_length_);
                }
            }

            // If sequence itself is too long, truncate it
            if (seq.size() > max_length_) {
                current.insert(current.end(), seq.begin(), seq.begin() + max_length_);
                packed.push_back(std::move(current));
                current = std::vector<int32_t>();
                current.reserve(max_length_);
                continue;
            }

            // Add separator if not first in chunk
            if (!current.empty()) {
                current.push_back(sep_token_id_);
            }

            // Add sequence
            current.insert(current.end(), seq.begin(), seq.end());
        }

        // Don't forget the last chunk
        if (!current.empty()) {
            packed.push_back(std::move(current));
        }

        return packed;
    }

    /**
     * @brief Calculate packing efficiency
     * @param sequences Input sequences
     * @return Ratio of actual tokens to total capacity
     */
    double efficiency(const std::vector<std::vector<int32_t>>& sequences) const {
        size_t total_tokens = 0;
        for (const auto& seq : sequences) {
            total_tokens += seq.size();
        }

        auto packed = pack(sequences);
        size_t total_capacity = packed.size() * max_length_;

        if (total_capacity == 0) return 0.0;
        return static_cast<double>(total_tokens) / total_capacity;
    }

private:
    size_t max_length_;
    int32_t sep_token_id_;
};

/**
 * @brief Text dataset loader
 *
 * Loads text from files and provides tokenized batches.
 */
class TextDataset {
public:
    TextDataset(const std::string& path, const Vocabulary& vocab,
                size_t max_length, bool add_bos = true, bool add_eos = true)
        : tokenizer_(vocab), batcher_(max_length, vocab.pad_id()),
          add_bos_(add_bos), add_eos_(add_eos) {

        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                auto ids = tokenizer_.encode(line, add_bos_, add_eos_);
                sequences_.push_back(std::move(ids));
            }
        }
    }

    size_t size() const { return sequences_.size(); }

    const std::vector<int32_t>& operator[](size_t idx) const {
        return sequences_[idx];
    }

    TokenizedSequence get(size_t idx) const {
        return batcher_.process(sequences_[idx]);
    }

    TokenizedBatch get_batch(size_t start, size_t batch_size) const {
        std::vector<std::vector<int32_t>> batch_seqs;
        for (size_t i = start; i < std::min(start + batch_size, sequences_.size()); ++i) {
            batch_seqs.push_back(sequences_[i]);
        }
        return batcher_.create_batch(batch_seqs);
    }

private:
    BasicTokenizer tokenizer_;
    SequenceBatcher batcher_;
    std::vector<std::vector<int32_t>> sequences_;
    bool add_bos_;
    bool add_eos_;
};

}  // namespace turboloader
