/**
 * @file test_text_decoder.cpp
 * @brief Tests for text/NLP data loading and tokenization
 */

#include <gtest/gtest.h>
#include "../src/decode/text_decoder.hpp"
#include <fstream>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

using namespace turboloader;

class TextDecoderTest : public ::testing::Test {
protected:
    std::string test_dir_;
    std::string test_file_;

    void SetUp() override {
        test_dir_ = "/tmp/turboloader_text_test_" + std::to_string(std::random_device{}());
        fs::create_directories(test_dir_);
        test_file_ = test_dir_ + "/corpus.txt";
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    void create_test_file(const std::vector<std::string>& lines) {
        std::ofstream file(test_file_);
        for (const auto& line : lines) {
            file << line << "\n";
        }
    }
};

// ============================================================================
// Vocabulary Tests
// ============================================================================

TEST_F(TextDecoderTest, VocabularyConstruction) {
    Vocabulary vocab;

    // Should have special tokens
    EXPECT_GT(vocab.size(), 0);
    EXPECT_NE(vocab.pad_id(), vocab.unk_id());
}

TEST_F(TextDecoderTest, VocabularySpecialTokens) {
    Vocabulary vocab;

    EXPECT_EQ(vocab.pad_id(), 0);
    EXPECT_EQ(vocab.unk_id(), 1);
    EXPECT_GE(vocab.bos_id(), 0);
    EXPECT_GE(vocab.eos_id(), 0);
}

TEST_F(TextDecoderTest, VocabularyAddToken) {
    Vocabulary vocab;

    int32_t id1 = vocab.add_token("hello");
    int32_t id2 = vocab.add_token("world");
    int32_t id3 = vocab.add_token("hello");  // Duplicate

    EXPECT_NE(id1, id2);
    EXPECT_EQ(id1, id3);  // Same token, same ID
}

TEST_F(TextDecoderTest, VocabularyGetId) {
    Vocabulary vocab;
    vocab.add_token("test");

    EXPECT_NE(vocab.get_id("test"), vocab.unk_id());
    EXPECT_EQ(vocab.get_id("nonexistent"), vocab.unk_id());
}

TEST_F(TextDecoderTest, VocabularyGetToken) {
    Vocabulary vocab;
    int32_t id = vocab.add_token("mytoken");

    EXPECT_EQ(vocab.get_token(id), "mytoken");
    EXPECT_EQ(vocab.get_token(99999), "[UNK]");  // Invalid ID
}

TEST_F(TextDecoderTest, VocabularyContains) {
    Vocabulary vocab;
    vocab.add_token("exists");

    EXPECT_TRUE(vocab.contains("exists"));
    EXPECT_FALSE(vocab.contains("missing"));
}

TEST_F(TextDecoderTest, VocabularyBuildFromTexts) {
    Vocabulary vocab;
    std::vector<std::string> texts = {
        "hello world",
        "hello there",
        "goodbye world"
    };
    vocab.build_from_texts(texts);

    EXPECT_TRUE(vocab.contains("hello"));
    EXPECT_TRUE(vocab.contains("world"));
    EXPECT_TRUE(vocab.contains("there"));
    EXPECT_TRUE(vocab.contains("goodbye"));
}

TEST_F(TextDecoderTest, VocabularyBuildFromFile) {
    create_test_file({"hello world", "hello there", "goodbye world"});

    Vocabulary vocab;
    vocab.build_from_file(test_file_);

    EXPECT_TRUE(vocab.contains("hello"));
    EXPECT_TRUE(vocab.contains("world"));
}

TEST_F(TextDecoderTest, VocabularyBuildWithMaxSize) {
    std::vector<std::string> texts = {
        "a a a a a",    // 'a' appears 5 times
        "b b b b",       // 'b' appears 4 times
        "c c c",         // 'c' appears 3 times
        "d d",           // 'd' appears 2 times
        "e"              // 'e' appears 1 time
    };

    Vocabulary vocab;
    vocab.build_from_texts(texts, 10);  // Allow 10 tokens (including special)

    // Most frequent tokens should be included
    EXPECT_TRUE(vocab.contains("a"));
    EXPECT_TRUE(vocab.contains("b"));
    EXPECT_TRUE(vocab.contains("c"));
}

TEST_F(TextDecoderTest, VocabularyBuildWithMinFreq) {
    std::vector<std::string> texts = {
        "a a a a a",
        "b",
        "c"
    };

    Vocabulary vocab;
    vocab.build_from_texts(texts, 0, 3);  // Min frequency 3

    EXPECT_TRUE(vocab.contains("a"));
    EXPECT_FALSE(vocab.contains("b"));  // Only appears once
    EXPECT_FALSE(vocab.contains("c"));
}

TEST_F(TextDecoderTest, VocabularySaveLoad) {
    Vocabulary vocab1;
    vocab1.add_token("hello");
    vocab1.add_token("world");

    std::string vocab_file = test_dir_ + "/vocab.txt";
    vocab1.save(vocab_file);

    Vocabulary vocab2;
    vocab2.load(vocab_file);

    EXPECT_EQ(vocab1.size(), vocab2.size());
    EXPECT_TRUE(vocab2.contains("hello"));
    EXPECT_TRUE(vocab2.contains("world"));
}

// ============================================================================
// BasicTokenizer Tests
// ============================================================================

TEST_F(TextDecoderTest, TokenizerEncode) {
    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("world");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("hello world");

    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(vocab.get_token(ids[0]), "hello");
    EXPECT_EQ(vocab.get_token(ids[1]), "world");
}

TEST_F(TextDecoderTest, TokenizerEncodeWithSpecialTokens) {
    Vocabulary vocab;
    vocab.add_token("hello");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("hello", true, true);  // Add BOS and EOS

    EXPECT_GE(ids.size(), 3);
    EXPECT_EQ(ids.front(), vocab.bos_id());
    EXPECT_EQ(ids.back(), vocab.eos_id());
}

TEST_F(TextDecoderTest, TokenizerDecode) {
    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("world");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("hello world");
    auto text = tokenizer.decode(ids);

    EXPECT_EQ(text, "hello world");
}

TEST_F(TextDecoderTest, TokenizerDecodeSkipSpecial) {
    Vocabulary vocab;
    vocab.add_token("hello");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("hello", true, true);
    auto text = tokenizer.decode(ids, true);  // Skip special tokens

    EXPECT_EQ(text, "hello");
}

TEST_F(TextDecoderTest, TokenizerHandlesPunctuation) {
    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token(",");
    vocab.add_token("world");
    vocab.add_token("!");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("Hello, world!");

    EXPECT_EQ(ids.size(), 4);
}

TEST_F(TextDecoderTest, TokenizerHandlesCase) {
    Vocabulary vocab;
    vocab.add_token("hello");  // Lowercase in vocab

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("HELLO");

    // Should lowercase before lookup
    EXPECT_EQ(vocab.get_token(ids[0]), "hello");
}

TEST_F(TextDecoderTest, TokenizerUnknownWords) {
    Vocabulary vocab;
    vocab.add_token("known");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("known unknown");

    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[1], vocab.unk_id());
}

// ============================================================================
// SequenceBatcher Tests
// ============================================================================

TEST_F(TextDecoderTest, BatcherPadding) {
    SequenceBatcher batcher(10, 0);

    std::vector<int32_t> short_seq = {1, 2, 3};
    auto result = batcher.process(short_seq);

    EXPECT_EQ(result.input_ids.size(), 10);
    EXPECT_EQ(result.input_ids[0], 1);
    EXPECT_EQ(result.input_ids[2], 3);
    EXPECT_EQ(result.input_ids[3], 0);  // Padding
}

TEST_F(TextDecoderTest, BatcherTruncation) {
    SequenceBatcher batcher(5, 0);

    std::vector<int32_t> long_seq = {1, 2, 3, 4, 5, 6, 7, 8};
    auto result = batcher.process(long_seq);

    EXPECT_EQ(result.input_ids.size(), 5);
    EXPECT_EQ(result.input_ids[4], 5);  // Truncated at right
}

TEST_F(TextDecoderTest, BatcherAttentionMask) {
    SequenceBatcher batcher(10, 0);

    std::vector<int32_t> seq = {1, 2, 3};
    auto result = batcher.process(seq);

    // First 3 should be 1, rest should be 0
    EXPECT_EQ(result.attention_mask[0], 1);
    EXPECT_EQ(result.attention_mask[2], 1);
    EXPECT_EQ(result.attention_mask[3], 0);
    EXPECT_EQ(result.attention_mask[9], 0);
}

TEST_F(TextDecoderTest, BatcherLeftPadding) {
    SequenceBatcher batcher(10, 0,
        SequenceBatcher::PaddingSide::LEFT,
        SequenceBatcher::TruncationSide::RIGHT);

    std::vector<int32_t> seq = {1, 2, 3};
    auto result = batcher.process(seq);

    // Padding should be on left
    EXPECT_EQ(result.input_ids[0], 0);
    EXPECT_EQ(result.input_ids[6], 0);
    EXPECT_EQ(result.input_ids[7], 1);
    EXPECT_EQ(result.input_ids[9], 3);
}

TEST_F(TextDecoderTest, BatcherLeftTruncation) {
    SequenceBatcher batcher(5, 0,
        SequenceBatcher::PaddingSide::RIGHT,
        SequenceBatcher::TruncationSide::LEFT);

    std::vector<int32_t> seq = {1, 2, 3, 4, 5, 6, 7, 8};
    auto result = batcher.process(seq);

    // Should keep last 5 elements
    EXPECT_EQ(result.input_ids[0], 4);
    EXPECT_EQ(result.input_ids[4], 8);
}

TEST_F(TextDecoderTest, BatcherCreateBatch) {
    SequenceBatcher batcher(5, 0);

    std::vector<std::vector<int32_t>> sequences = {
        {1, 2, 3},
        {4, 5},
        {6, 7, 8, 9, 10, 11}  // Will be truncated
    };

    auto batch = batcher.create_batch(sequences);

    EXPECT_EQ(batch.batch_size(), 3);
    EXPECT_EQ(batch.max_length(), 5);
}

TEST_F(TextDecoderTest, BatcherTokenTypeIds) {
    SequenceBatcher batcher(10, 0);

    std::vector<int32_t> seq = {1, 2, 3};
    auto result = batcher.process(seq);

    // Token type IDs should be initialized to 0
    for (int32_t id : result.token_type_ids) {
        EXPECT_EQ(id, 0);
    }
}

// ============================================================================
// SequencePacker Tests
// ============================================================================

TEST_F(TextDecoderTest, PackerBasic) {
    SequencePacker packer(10, 99);  // max_length=10, sep=99

    std::vector<std::vector<int32_t>> sequences = {
        {1, 2, 3},     // 3 tokens
        {4, 5, 6},     // 3 tokens
        {7, 8}         // 2 tokens
    };
    // Total: 8 tokens + 2 separators = 10, fits in one chunk

    auto packed = packer.pack(sequences);

    EXPECT_EQ(packed.size(), 1);
    EXPECT_EQ(packed[0].size(), 10);  // 3 + 1 + 3 + 1 + 2
}

TEST_F(TextDecoderTest, PackerMultipleChunks) {
    SequencePacker packer(10, 99);

    std::vector<std::vector<int32_t>> sequences = {
        {1, 2, 3, 4, 5},  // 5 tokens
        {6, 7, 8},         // 3 tokens + sep = 4, total 9
        {9, 10}            // Doesn't fit, new chunk
    };

    auto packed = packer.pack(sequences);

    EXPECT_GE(packed.size(), 2);
}

TEST_F(TextDecoderTest, PackerLongSequence) {
    SequencePacker packer(5, 99);

    std::vector<std::vector<int32_t>> sequences = {
        {1, 2, 3, 4, 5, 6, 7, 8}  // Longer than max_length
    };

    auto packed = packer.pack(sequences);

    // Should be truncated to 5
    EXPECT_EQ(packed[0].size(), 5);
}

TEST_F(TextDecoderTest, PackerEfficiency) {
    SequencePacker packer(100, 99);

    std::vector<std::vector<int32_t>> sequences;
    for (int i = 0; i < 10; ++i) {
        sequences.push_back({1, 2, 3, 4, 5});  // Each sequence is 5 tokens
    }

    double efficiency = packer.efficiency(sequences);
    EXPECT_GE(efficiency, 0.5);  // Should be reasonably efficient
}

// ============================================================================
// TextDataset Tests
// ============================================================================

TEST_F(TextDecoderTest, DatasetConstruction) {
    create_test_file({"hello world", "goodbye world"});

    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("goodbye");
    vocab.add_token("world");

    TextDataset dataset(test_file_, vocab, 10);

    EXPECT_EQ(dataset.size(), 2);
}

TEST_F(TextDecoderTest, DatasetAccess) {
    create_test_file({"hello world"});

    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("world");

    TextDataset dataset(test_file_, vocab, 10);

    auto seq = dataset[0];
    EXPECT_GT(seq.size(), 0);
}

TEST_F(TextDecoderTest, DatasetGet) {
    create_test_file({"hello world"});

    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("world");

    TextDataset dataset(test_file_, vocab, 10);

    auto processed = dataset.get(0);
    EXPECT_EQ(processed.input_ids.size(), 10);
    EXPECT_EQ(processed.attention_mask.size(), 10);
}

TEST_F(TextDecoderTest, DatasetGetBatch) {
    create_test_file({"line 1", "line 2", "line 3", "line 4"});

    Vocabulary vocab;
    vocab.add_token("line");
    vocab.add_token("1");
    vocab.add_token("2");
    vocab.add_token("3");
    vocab.add_token("4");

    TextDataset dataset(test_file_, vocab, 10);

    auto batch = dataset.get_batch(0, 2);
    EXPECT_EQ(batch.batch_size(), 2);
}

// ============================================================================
// Special Tokens Tests
// ============================================================================

TEST_F(TextDecoderTest, CustomSpecialTokens) {
    SpecialTokens special;
    special.pad_token = "<pad>";
    special.unk_token = "<unk>";
    special.bos_token = "<s>";
    special.eos_token = "</s>";

    Vocabulary vocab(special);

    EXPECT_TRUE(vocab.contains("<pad>"));
    EXPECT_TRUE(vocab.contains("<unk>"));
    EXPECT_TRUE(vocab.contains("<s>"));
    EXPECT_TRUE(vocab.contains("</s>"));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TextDecoderTest, EmptyText) {
    Vocabulary vocab;
    BasicTokenizer tokenizer(vocab);

    auto ids = tokenizer.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST_F(TextDecoderTest, WhitespaceOnly) {
    Vocabulary vocab;
    BasicTokenizer tokenizer(vocab);

    auto ids = tokenizer.encode("   \t\n  ");
    EXPECT_TRUE(ids.empty());
}

TEST_F(TextDecoderTest, PunctuationOnly) {
    Vocabulary vocab;
    vocab.add_token(".");
    vocab.add_token(",");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode(".,.,.");

    EXPECT_EQ(ids.size(), 5);
}

TEST_F(TextDecoderTest, MixedUnicode) {
    // Basic ASCII handling (we don't fully support unicode in this simple impl)
    Vocabulary vocab;
    vocab.add_token("hello");

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode("hello");

    EXPECT_EQ(ids.size(), 1);
}

TEST_F(TextDecoderTest, VeryLongSequence) {
    Vocabulary vocab;
    for (int i = 0; i < 100; ++i) {
        vocab.add_token("word" + std::to_string(i));
    }

    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += "word" + std::to_string(i) + " ";
    }

    BasicTokenizer tokenizer(vocab);
    auto ids = tokenizer.encode(text);

    EXPECT_EQ(ids.size(), 100);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
