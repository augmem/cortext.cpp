// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/llm_executor_processed_tokens.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(ProcessedTokensTest, TokenEmpty) {
  ProcessedTokens processed_tokens;
  EXPECT_EQ(processed_tokens.TokenCount(), 0);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 0);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(0).empty());
}

TEST(ProcessedTokensTest, AddProcessedTokens) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(3).empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), std::vector<int>{4});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken_Failed) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_THAT(
      processed_tokens.AddPendingInputToken(
          {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5)}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken_AlreadyExists) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  EXPECT_THAT(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(5)}),
      StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
}

TEST(ProcessedTokensTest, RollBackToStep) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_OK(processed_tokens.RollBackToStep(1));
  EXPECT_EQ(processed_tokens.TokenCount(), 1);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 1);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(1).empty());
}

TEST(ProcessedTokensTest, RollBackToStep_WithPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  EXPECT_OK(processed_tokens.RollBackToStep(1));
  EXPECT_EQ(processed_tokens.TokenCount(), 1);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 1);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(1).empty());
}

TEST(ProcessedTokensTest, MarkPendingInputTokenAsProcessed) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  EXPECT_OK(processed_tokens.MarkPendingInputTokenAsProcessed());
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), std::vector<int>{4});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, MarkPendingInputTokenAsProcessed_NoPendingToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_THAT(processed_tokens.MarkPendingInputTokenAsProcessed(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(3).empty());
}

TEST(ProcessedTokensTest, GetCopyOfTokens) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_THAT(processed_tokens.GetCopyOfTokens(),
              (std::vector<std::vector<int>>{{1, 2, 3}}));
}

TEST(ProcessedTokensTest, GetCopyOfTokens_WithPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_THAT(processed_tokens.GetCopyOfTokens(),
              (std::vector<std::vector<int>>{{1, 2, 3, 4}}));
}

TEST(ProcessedTokensTest, InvalidPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  processed_tokens.InvalidatePendingInputToken();
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(3).empty());
}

TEST(ProcessedTokensTest, BroadcastTokenCandidates) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(3).empty());
}

TEST(ProcessedTokensTest, BroadcastTokenCandidates_WithPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 4);
  EXPECT_EQ(step_and_token.token[2]->id(), 4);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 4, 4}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, AddProcessedTokens_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  processed_tokens.AddProcessedTokens({4, 5});
  EXPECT_EQ(processed_tokens.TokenCount(), 5);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 5);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 4, 4}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(4), (std::vector<int>{5, 5, 5}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(5).empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 5, 6}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken_MultipleBatches_Failed) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_THAT(
      processed_tokens.AddPendingInputToken(
          {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5)}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
}

TEST(ProcessedTokensTest, AddPendingInputToken_MultipleBatches_AlreadyExists) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  EXPECT_THAT(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(7),
                                             std::make_shared<TokenData>(8),
                                             std::make_shared<TokenData>(9)}),
      StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 5, 6}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, RollBackToStep_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());

  EXPECT_OK(processed_tokens.RollBackToStep(1));
  EXPECT_EQ(processed_tokens.TokenCount(), 1);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 1);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(1).empty());
}

TEST(ProcessedTokensTest,
     RollBackToStep_MultipleBatches_WithPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(
      processed_tokens.AddPendingInputToken({std::make_shared<TokenData>(4)}));
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 4);
  EXPECT_EQ(step_and_token.token[2]->id(), 4);

  EXPECT_OK(processed_tokens.RollBackToStep(1));
  EXPECT_EQ(processed_tokens.TokenCount(), 1);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 1);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(1).empty());
}

TEST(ProcessedTokensTest, MarkPendingInputTokenAsProcessed_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  EXPECT_OK(processed_tokens.MarkPendingInputTokenAsProcessed());
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 5, 6}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, GetCopyOfTokens_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_THAT(processed_tokens.GetCopyOfTokens(),
              (std::vector<std::vector<int>>{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}));
}

TEST(ProcessedTokensTest,
     GetCopyOfTokens_MultipleBatches_WithPendingInputToken) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_THAT(processed_tokens.GetCopyOfTokens(),
              (std::vector<std::vector<int>>{
                  {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 3, 6}}));
}

TEST(ProcessedTokensTest, InvalidatePendingInputToken_MultipleBatches) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  processed_tokens.InvalidatePendingInputToken();
  EXPECT_EQ(processed_tokens.TokenCount(), 3);
  step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_TRUE(step_and_token.token.empty());
  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(3).empty());
}

TEST(ProcessedTokensTest, ReduceTokenCandidates) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_OK(processed_tokens.ReduceTokenCandidates(0));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), std::vector<int>{4});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, ReduceTokenCandidates_2ndIndex) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_OK(processed_tokens.ReduceTokenCandidates(1));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 5);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), std::vector<int>{1});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), std::vector<int>{2});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), std::vector<int>{3});
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), std::vector<int>{5});
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

TEST(ProcessedTokensTest, ReduceTokenCandidates_OutOfRange) {
  ProcessedTokens processed_tokens;
  processed_tokens.AddProcessedTokens({1, 2, 3});
  EXPECT_OK(processed_tokens.BroadcastTokenCandidates(3));
  EXPECT_OK(processed_tokens.AddPendingInputToken(
      {std::make_shared<TokenData>(4), std::make_shared<TokenData>(5),
       std::make_shared<TokenData>(6)}));
  EXPECT_THAT(processed_tokens.ReduceTokenCandidates(4),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_EQ(processed_tokens.TokenCount(), 4);
  auto step_and_token = processed_tokens.GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 3);
  EXPECT_EQ(step_and_token.token.size(), 3);
  EXPECT_EQ(step_and_token.token[0]->id(), 4);
  EXPECT_EQ(step_and_token.token[1]->id(), 5);
  EXPECT_EQ(step_and_token.token[2]->id(), 6);

  EXPECT_THAT(processed_tokens.GetTokenAtStep(0), (std::vector<int>{1, 1, 1}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(1), (std::vector<int>{2, 2, 2}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(2), (std::vector<int>{3, 3, 3}));
  EXPECT_THAT(processed_tokens.GetTokenAtStep(3), (std::vector<int>{4, 5, 6}));
  EXPECT_TRUE(processed_tokens.GetTokenAtStep(4).empty());
}

}  // namespace
}  // namespace litert::lm
