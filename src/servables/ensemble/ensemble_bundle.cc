// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servables/ensemble/ensemble_bundle.h"

#include <stdint.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

tensorflow::Status
EnsembleBundle::Init(
    const tensorflow::StringPiece& path, const ModelConfig& config,
    uint64_t inference_server)
{
  TF_RETURN_IF_ERROR(ValidateModelConfig(config, kEnsemblePlatform));
  TF_RETURN_IF_ERROR(SetModelConfig(path, config));

  // [TODO] Replace it to ensemble scheduler after DLIS-290
  TF_RETURN_IF_ERROR(SetConfiguredScheduler(
      1,
      [](uint32_t runner_idx) -> tensorflow::Status {
        return tensorflow::Status::OK();
      },
      [this](
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(tensorflow::Status)> func) {
        Run(runner_idx, payloads, func);
      }));

  inference_server_ = inference_server;

  LOG_VERBOSE(1) << "ensemble bundle for " << Name() << std::endl << *this;

  return tensorflow::Status::OK();
}

void
EnsembleBundle::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads)
{
  LOG_ERROR << "Unexpectedly invoked EnsembleBundle::Run()";

  OnCompleteQueuedPayloads(tensorflow::errors::Internal(
      "unexpected invocation of EnsembleBundle::Run()"));
}

std::ostream&
operator<<(std::ostream& out, const EnsembleBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  return out;
}

}}  // namespace nvidia::inferenceserver
