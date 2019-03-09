// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>
#include <unistd.h>
#include <csignal>
#include <thread>

#include "src/core/grpc_server.h"
#include "src/core/http_server.h"
#include "src/core/logging.h"
#include "src/core/server.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {

// The inference server object. Once this server is successfully
// created it does *not* transition back to a nullptr value and it is
// *not* explicitly destructed. Thus we assume that 'server_' can
// always be dereferenced.
nvidia::inferenceserver::InferenceServer* server_ = nullptr;

// Thread used to close and exit the server.
std::unique_ptr<std::thread> exit_thread_;

// The HTTP and GRPC services
std::unique_ptr<nvidia::inferenceserver::HTTPServer> http_service_;
std::unique_ptr<nvidia::inferenceserver::GRPCServer> grpc_service_;

// The HTTP and GRPC ports. Initialized to default values and
// modifyied based on command-line args. Set to -1 to indicate the
// protocol is disabled.
int http_port_ = 8000;
int grpc_port_ = 8001;

// The number of threads to initialize for the HTTP front-end.
int http_thread_cnt_ = 8;

void
SignalHandler(int signum)
{
  // Don't need a mutex here since signals should be disabled while in
  // the handler.
  LOG_INFO << "Interrupt signal (" << signum << ") received.";

  if (exit_thread_ != nullptr)
    return;

  exit_thread_.reset(new std::thread([] {
    bool close_status = server_->Close();
    exit(close_status ? 0 : 1);
  }));

  exit_thread_->detach();
}

std::unique_ptr<nvidia::inferenceserver::GRPCServer>
StartGrpcService(nvidia::inferenceserver::InferenceServer* server)
{
  std::unique_ptr<nvidia::inferenceserver::GRPCServer> service;
  tensorflow::Status status =
      nvidia::inferenceserver::GRPCServer::Create(server, grpc_port_, &service);
  if (status.ok()) {
    status = service->Start();
  }

  if (!status.ok()) {
    service.reset();
  }

  return std::move(service);
}

std::unique_ptr<nvidia::inferenceserver::HTTPServer>
StartHttpService(nvidia::inferenceserver::InferenceServer* server)
{
  std::unique_ptr<nvidia::inferenceserver::HTTPServer> service;
  tensorflow::Status status = nvidia::inferenceserver::HTTPServer::Create(
      server, http_port_, http_thread_cnt_, &service);
  if (status.ok()) {
    status = service->Start();
  }

  if (!status.ok()) {
    service.reset();
  }

  return std::move(service);
}

bool
StartEndpoints(nvidia::inferenceserver::InferenceServer* server)
{
  LOG_INFO << "Starting endpoints, '" << server->Id() << "' listening on";

  // Enable gRPC endpoint if requested...
  if (grpc_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(grpc_port_)
             << " for gRPC requests";
    grpc_service_ = StartGrpcService(server);
    if (grpc_service_ == nullptr) {
      LOG_ERROR << "Failed to start gRPC service";
      return false;
    }
  }

  // Enable HTTP endpoint if requested...
  if (http_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(http_port_)
             << " for HTTP requests";

    http_service_ = StartHttpService(server);
    if (http_service_ == nullptr) {
      LOG_ERROR << "Failed to start HTTP service";
      return false;
    }
  }

  return true;
}

bool
Parse(nvidia::inferenceserver::InferenceServer* server, int argc, char** argv)
{
  std::string server_id(server->Id());
  std::string model_store_path(server->ModelStorePath());
  std::string platform_config_file(server->PlatformConfigFile());
  bool strict_model_config = server->StrictModelConfigEnabled();
  bool strict_readiness = server->StrictReadinessEndabled();
  bool allow_profiling = server->ProfilingEnabled();
  bool tf_allow_soft_placement = server->TensorFlowSoftPlacementEnabled();
  float tf_gpu_memory_fraction = server->TensorFlowGPUMemoryFraction();
  bool allow_poll_model_repository = server->PollModelRepositoryEnabled();
  int32_t repository_poll_secs = server->RepositoryPollSeconds();
  int32_t exit_timeout_secs = server->ExitTimeoutSeconds();

  bool exit_on_error = true;
  bool allow_http = true;
  bool allow_grpc = true;
  bool allow_metrics = true;
  int32_t http_port = http_port_;
  int32_t grpc_port = grpc_port_;
  int32_t metrics_port = metrics_port_;
  int32_t http_thread_cnt = http_thread_cnt_;

  bool log_info = true;
  bool log_warn = true;
  bool log_error = true;
  int32_t log_verbose = 0;

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("log-info", &log_info, "Enable/Disable info logging"),
      tensorflow::Flag(
          "log-warning", &log_warn, "Enable/Disable warning logging"),
      tensorflow::Flag("log-error", &log_error, "Enable/Disable error logging"),
      tensorflow::Flag("log-verbose", &log_verbose, "Verbose logging level"),
      tensorflow::Flag("id", &server_id, "Identifier for this server"),
      tensorflow::Flag(
          "model-store", &model_store_path, "Path to model store directory."),
      tensorflow::Flag(
          "platform-config-file", &platform_config_file,
          "If non-empty, read an ASCII PlatformConfigMap protobuf "
          "from the supplied file name, and use that platform "
          "config instead of the default platform."),
      tensorflow::Flag(
          "exit-on-error", &exit_on_error,
          "Exit the inference server if an error occurs during "
          "initialization."),
      tensorflow::Flag(
          "strict-model-config", &strict_model_config,
          "If true model configuration files must be provided and all required "
          "configuration settings must be specified. If false the model "
          "configuration may be absent or only partially specified and the "
          "server will attempt to derive the missing required configuration."),
      tensorflow::Flag(
          "strict-readiness", &strict_readiness,
          "If true /api/health/ready endpoint indicates ready if the server "
          "is responsive and all models are available. If false "
          "/api/health/ready endpoint indicates ready if server is responsive "
          "even "
          "if some/all models are unavailable."),
      tensorflow::Flag(
          "allow-profiling", &allow_profiling, "Allow server profiling."),
      tensorflow::Flag(
          "allow-http", &allow_http,
          "Allow the server to listen on for HTTP requests."),
      tensorflow::Flag(
          "allow-grpc", &allow_grpc,
          "Allow the server to listen on for gRPC requests."),
      tensorflow::Flag(
          "allow-metrics", &allow_metrics,
          "Allow the server to provide prometheus metrics."),
      tensorflow::Flag(
          "http-port", &http_port,
          "The port for the server to listen on for HTTP requests."),
      tensorflow::Flag(
          "grpc-port", &grpc_port,
          "The port for the server to listen on for gRPC requests."),
      tensorflow::Flag(
          "metrics-port", &metrics_port,
          "The port exposing prometheus metrics."),
      tensorflow::Flag(
          "http-thread-count", &http_thread_cnt,
          "Number of threads handling HTTP requests."),
      tensorflow::Flag(
          "allow-poll-model-repository", &allow_poll_model_repository,
          "Poll the model repository to detect changes. The poll rate is "
          "controlled by 'repository-poll-secs'."),
      tensorflow::Flag(
          "repository-poll-secs", &repository_poll_secs,
          "Interval in seconds between each poll of the model repository to "
          "check "
          "for changes. A value of zero indicates that the repository is "
          "checked "
          "only a single time at startup. Valid only when "
          "--allow-poll-model-repository=true is specified."),
      tensorflow::Flag(
          "exit-timeout-secs", &exit_timeout_secs,
          "Timeout (in seconds) when exiting to wait for in-flight inferences "
          "to "
          "finish. After the timeout expires the server exits even if "
          "inferences "
          "are still in flight."),
      tensorflow::Flag(
          "tf-allow-soft-placement", &tf_allow_soft_placement,
          "Instruct TensorFlow to use CPU implementation of an operation when "
          "a GPU implementation is not available."),
      tensorflow::Flag(
          "tf-gpu-memory-fraction", &tf_gpu_memory_fraction,
          "Reserve a portion of GPU memory for TensorFlow models. Default "
          "value 0.0 indicates that TensorFlow should dynamically allocate "
          "memory as needed. Value of 1.0 indicates that TensorFlow should "
          "allocate all of GPU memory."),
  };

  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LogInitError(usage);
    return false;
  }

  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);

  http_port_ = allow_http ? http_port : -1;
  grpc_port_ = allow_grpc ? grpc_port : -1;
  http_thread_cnt_ = http_thread_cnt;

  server->SetId(server_id);
  server->SetModelStorePath(model_store_path);
  server->SetPlatformConfigFile(platform_config_file);
  server->SetStrictModelConfigEnabled(strict_model_config);
  server->SetStrictReadinessEnabled(strict_readiness);
  server->SetProfilingEnabled(allow_profiling);
  server->SetPollModelRepositoryEnabled(allow_poll_model_repository);
  server->SetRepositoryPollSeconds(std::max(0, repository_poll_secs));
  server->SetExitTimeoutSeconds(std::max(0, exit_timeout_secs));
  server->SetTensorFlowAllowSoftPlacement(tf_allow_soft_placement);
  server->SetTensorFlowGPUMemoryFraction(tf_gpu_memory_fraction);

  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  // Create the inference server
  server_ = new nvidia::inferenceserver::InferenceServer();

  // Parse command-line using defaults provided by the inference
  // server. Update inference server options appropriately.
  if (!Parse(server_, argc, argv)) {
    exit(1);
  }

  // Initialize the inference server
  if (!server_->Init()) {
    exit(1);
  }

  // Start the HTTP and/or GRPC endpoints.
  if (!StartEndpoints(server_)) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Watch for changes in the model repository.
  server_->WatchModelRepository();

  // The main thread doesn't exit. The inference server is only
  // stopped by sending it a signal and then the exit_thread_ in the
  // signal handler exits the process. I don't believe these stop
  // methods don't actually return...
  if (grpc_service_) {
    grpc_service_->Stop();
  }
  if (http_service_ != nullptr) {
    http_service_->Stop();
  }

  return 0;
}
