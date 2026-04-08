// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_BASE_H_

#include <atomic>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/resource_manager.h"

namespace litert::lm {

struct SessionInfo {
  SessionConfig session_config;
  std::shared_ptr<ContextHandler> context_handler;
  std::unique_ptr<Sampler> sampler;
  std::unique_ptr<StopTokenDetector> stop_token_detector;
  std::optional<BenchmarkInfo> benchmark_info;
  absl::flat_hash_set<TaskId> active_tasks = {};
  int last_prefill_token_id = 0;
};

struct TaskInfo {
  SessionId session_id;
  TaskState task_state;
  absl::AnyInvocable<void()> task;
  absl::flat_hash_set<TaskId> dependent_tasks = {};
  absl::flat_hash_set<TaskId> following_tasks = {};
  std::shared_ptr<std::atomic<bool>> cancelled;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback;
};

class ExecutionManagerBase : public ExecutionManager {
 public:
  ~ExecutionManagerBase() override = default;

  absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info = std::nullopt) override = 0;

  absl::Status ReleaseSession(SessionId session_id) override = 0;

  absl::Status CancelAllTasksInSession(SessionId session_id) override = 0;

  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) override = 0;

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(
      SessionId session_id) override = 0;

  absl::StatusOr<TaskId> GetNewTaskId() override = 0;

  absl::Status AddPrefillTask(
      SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      int max_output_tokens) override;

  absl::Status AddCloneSessionTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks, SessionId cloned_session_id,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::Status AddTextScoringTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::StatusOr<int> GetCurrentStep(const SessionInfo& session_info) override = 0;

  absl::Status SetCurrentStep(const SessionInfo& session_info,
                              int target_step) override = 0;

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override;

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override;

 protected:
  ExecutionManagerBase(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr)
      : tokenizer_(std::move(tokenizer)),
        resource_manager_(std::move(resource_manager)),
        litert_env_(litert_env) {}

  // Unlocked implementation methods.
  absl::StatusOr<std::shared_ptr<SessionInfo>> CreateSessionInfo(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info);

  absl::StatusOr<SessionId> RegisterNewSessionUnlocked(
      std::shared_ptr<SessionInfo> session_info);

  absl::Status ReleaseSessionUnlocked(SessionId session_id);

  absl::Status CancelAllTasksInSessionUnlocked(SessionId session_id);

  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfoUnlocked(
      SessionId session_id);

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfoUnlocked(
      SessionId session_id);

  absl::Status CreateTaskUnlocked(
      SessionId session_id, TaskId task_id,
      absl::AnyInvocable<void()> task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback);

  absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTaskUnlocked(TaskId task_id);

  absl::Status FinishTaskUnlocked(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback);

  absl::StatusOr<absl::flat_hash_set<TaskId>> FollowingWaitingTasksUnlocked(
      TaskId task_id);

  absl::Status UpdateTaskStateUnlocked(TaskId task_id, TaskState task_state);

  absl::Status UpdateAllTasksToStateUnlocked(
      const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state);

  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& preprocessed_contents,
      std::optional<BenchmarkInfo>& benchmark_info);

  absl::StatusOr<int> GetCurrentStepUnlocked(const SessionInfo& session_info);
  absl::Status SetCurrentStepUnlocked(const SessionInfo& session_info,
                                      int target_step);

  virtual absl::Status CreateTask(
      SessionId session_id, TaskId task_id,
      absl::AnyInvocable<void()> task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  virtual absl::Status QueueTask(TaskId task_id) = 0;

  virtual absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) = 0;

  virtual absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) = 0;

  virtual void FinishTaskAndLogErrors(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) = 0;

  std::atomic<SessionId> next_session_id_ = 0;
  std::atomic<TaskId> next_task_id_ = 0;

  absl::flat_hash_map<SessionId, std::shared_ptr<SessionInfo> absl_nonnull>
      session_lookup_ = {};
  absl::flat_hash_map<TaskId, TaskInfo> task_lookup_ = {};

  int last_prefill_token_id_ = 0;

  Tokenizer* absl_nonnull tokenizer_;
  std::unique_ptr<ResourceManager> absl_nonnull resource_manager_;
  ::litert::Environment* absl_nullable litert_env_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_BASE_H_
