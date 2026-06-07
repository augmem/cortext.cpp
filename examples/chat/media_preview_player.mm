#include "media_preview_player.hpp"

#import <AVFoundation/AVFoundation.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace chat {

struct MediaPreviewPlayer::Impl {
  Impl() {
    @autoreleasepool {
      engine = [[AVAudioEngine alloc] init];
      player = [[AVAudioPlayerNode alloc] init];
      [engine attachNode:player];
      AVAudioMixerNode* mixer = [engine mainMixerNode];
      output_format = [[mixer outputFormatForBus:0] retain];
      [engine connect:player to:mixer format:output_format];
      NSError* error = nil;
      available = [engine startAndReturnError:&error];
    }
  }

  ~Impl() {
    Stop();
    if (worker.joinable()) {
      worker.join();
    }
    @autoreleasepool {
      if (player != nil) {
        [player release];
        player = nil;
      }
      if (engine != nil) {
        [engine stop];
        [engine release];
        engine = nil;
      }
      if (output_format != nil) {
        [output_format release];
        output_format = nil;
      }
    }
  }

  void Play(std::vector<float> samples, std::int32_t sample_rate) {
    Stop();
    if (worker.joinable()) {
      worker.join();
    }
    cancelled.store(false);
    playing.store(true);
    worker = std::thread([this, samples = std::move(samples), sample_rate] {
      PlayBlocking(samples, sample_rate);
      playing.store(false);
    });
  }

  void Stop() {
    cancelled.store(true);
    @autoreleasepool {
      if (player != nil) {
        [player stop];
      }
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      done = true;
    }
    cv.notify_all();
    playing.store(false);
  }

  void PlayBlocking(const std::vector<float>& samples, std::int32_t sample_rate) {
    if (!available || samples.empty() || sample_rate <= 0 || player == nil
        || output_format == nil) {
      return;
    }

    @autoreleasepool {
      std::unique_lock<std::mutex> lock(mu);
      done = false;

      AVAudioFormat* source_format = [[[AVAudioFormat alloc]
          initWithCommonFormat:AVAudioPCMFormatFloat32
                    sampleRate:sample_rate
                      channels:1
                   interleaved:NO] autorelease];
      AVAudioPCMBuffer* source_buffer = [[[AVAudioPCMBuffer alloc]
          initWithPCMFormat:source_format
              frameCapacity:static_cast<AVAudioFrameCount>(samples.size())] autorelease];
      source_buffer.frameLength = static_cast<AVAudioFrameCount>(samples.size());
      std::memcpy(source_buffer.floatChannelData[0], samples.data(),
                  samples.size() * sizeof(float));

      AVAudioConverter* converter = [[[AVAudioConverter alloc]
          initFromFormat:source_format
                 toFormat:output_format] autorelease];
      if (converter == nil) {
        return;
      }

      const double ratio = output_format.sampleRate
                           / std::max(1.0, static_cast<double>(sample_rate));
      const AVAudioFrameCount target_frames
          = static_cast<AVAudioFrameCount>(std::ceil(samples.size() * ratio)) + 16;
      AVAudioPCMBuffer* output_buffer = [[[AVAudioPCMBuffer alloc]
          initWithPCMFormat:output_format
              frameCapacity:target_frames] autorelease];
      __block bool consumed = false;
      AVAudioConverterInputBlock input_block
          = ^AVAudioBuffer* (AVAudioPacketCount inNumberOfPackets,
                             AVAudioConverterInputStatus* outStatus) {
        (void)inNumberOfPackets;
        if (consumed) {
          *outStatus = AVAudioConverterInputStatus_NoDataNow;
          return nil;
        }
        consumed = true;
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return source_buffer;
      };

      NSError* error = nil;
      const AVAudioConverterOutputStatus status
          = [converter convertToBuffer:output_buffer
                                 error:&error
                    withInputFromBlock:input_block];
      if (status == AVAudioConverterOutputStatus_Error || error != nil
          || output_buffer.frameLength == 0) {
        return;
      }

      __block Impl* weak_self = this;
      [player stop];
      [player scheduleBuffer:output_buffer completionHandler:^{
        if (weak_self == nullptr) {
          return;
        }
        std::lock_guard<std::mutex> guard(weak_self->mu);
        weak_self->done = true;
        weak_self->cv.notify_all();
      }];
      [player play];
      cv.wait(lock, [&] {
        return done || cancelled.load();
      });
      if (cancelled.load()) {
        [player stop];
      }
    }
  }

  AVAudioEngine* engine = nil;
  AVAudioPlayerNode* player = nil;
  AVAudioFormat* output_format = nil;
  std::thread worker;
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> cancelled{false};
  std::atomic<bool> playing{false};
  bool available = false;
  bool done = false;
};

MediaPreviewPlayer::MediaPreviewPlayer() : impl_(std::make_unique<Impl>()) {}

MediaPreviewPlayer::~MediaPreviewPlayer() = default;

bool MediaPreviewPlayer::IsAvailable() const {
  return impl_ && impl_->available;
}

void MediaPreviewPlayer::PlayPcmF32(std::vector<float> samples,
                                    std::int32_t sample_rate) {
  if (impl_) {
    impl_->Play(std::move(samples), sample_rate);
  }
}

void MediaPreviewPlayer::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

bool MediaPreviewPlayer::IsPlaying() const {
  return impl_ && impl_->playing.load();
}

}  // namespace chat
