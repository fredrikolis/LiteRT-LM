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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

using SessionId = int;
using TaskId = int;

// Fwd declarations
class ContextHandler;
class Sampler;
class StopTokenDetector;
struct SessionInfo;
class Constraint;

// The execution manager interface.
class ExecutionManager {
 public:
  virtual ~ExecutionManager() = default;

  virtual absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout) = 0;
  virtual absl::Status WaitUntilSessionDone(SessionId session_id,
                                            absl::Duration timeout) = 0;
  virtual absl::Status WaitUntilAllDone(absl::Duration timeout) = 0;

  virtual absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info = std::nullopt) = 0;

  virtual absl::Status ReleaseSession(SessionId session_id) = 0;

  virtual absl::Status CancelAllTasksInSession(SessionId session_id) = 0;

  virtual absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) = 0;

  virtual absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(
      SessionId session_id) = 0;

  virtual absl::StatusOr<TaskId> GetNewTaskId() = 0;

  virtual absl::Status AddPrefillTask(
      SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  virtual absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      int max_output_tokens) = 0;

  virtual absl::Status AddCloneSessionTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks, SessionId cloned_session_id,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  virtual absl::Status AddTextScoringTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  virtual absl::StatusOr<int> GetCurrentStep(const SessionInfo& session_info) = 0;

  virtual absl::Status SetCurrentStep(const SessionInfo& session_info,
                                      int target_step) = 0;

  virtual absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const = 0;

  virtual absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
