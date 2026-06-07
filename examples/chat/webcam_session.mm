#include "webcam_session.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chat {

namespace {

std::string NSErrorMessage(NSError* error, const char* prefix) {
  std::string message = prefix;
  if (error != nil && error.localizedDescription != nil) {
    message += ": ";
    message += [[error localizedDescription] UTF8String];
  }
  return message;
}

bool RequestAccessSync(AVMediaType media_type) {
  const AVAuthorizationStatus status
      = [AVCaptureDevice authorizationStatusForMediaType:media_type];
  if (status == AVAuthorizationStatusAuthorized) {
    return true;
  }
  if (status == AVAuthorizationStatusDenied
      || status == AVAuthorizationStatusRestricted) {
    return false;
  }

  __block BOOL granted = NO;
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  [AVCaptureDevice requestAccessForMediaType:media_type
                           completionHandler:^(BOOL ok) {
    granted = ok;
    dispatch_semaphore_signal(sema);
  }];
  dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
#if !OS_OBJECT_USE_OBJC
  dispatch_release(sema);
#endif
  return granted == YES;
}

}  // namespace

}  // namespace chat

@interface CortextWebcamVideoDelegate : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate> {
 @public
  void* owner_;
}
- (instancetype)initWithOwner:(void*)owner;
@end

namespace chat {

struct WebcamSession::Impl {
  explicit Impl(WebcamSessionConfig cfg) : config(std::move(cfg)) {
    available = true;
  }

  ~Impl() {
    Stop();
  }

  bool Start() {
    std::lock_guard<std::mutex> lock(mu);
    if (capturing) {
      return true;
    }
    if (!available) {
      ReportError("Webcam capture is unavailable.");
      return false;
    }
    if (!RequestAccessSync(AVMediaTypeVideo)) {
      ReportError("Camera permission was not granted.");
      return false;
    }
    if (!RequestAccessSync(AVMediaTypeAudio)) {
      ReportError("Microphone permission was not granted.");
      return false;
    }
    if (!StartVideoLocked()) {
      StopLocked();
      return false;
    }
    if (!StartAudioLocked()) {
      StopLocked();
      return false;
    }
    capturing = true;
    if (config.on_capture_changed) {
      config.on_capture_changed(true);
    }
    return true;
  }

  void Stop() {
    std::lock_guard<std::mutex> lock(mu);
    StopLocked();
  }

  void StopLocked() {
    if (video_session != nil) {
      [video_session stopRunning];
    }
    if (video_output != nil) {
      [video_output setSampleBufferDelegate:nil queue:nil];
    }
    if (video_queue != nullptr) {
#if !OS_OBJECT_USE_OBJC
      dispatch_release(video_queue);
#endif
      video_queue = nullptr;
    }
    if (video_delegate != nil) {
      [video_delegate release];
      video_delegate = nil;
    }
    if (video_output != nil) {
      [video_output release];
      video_output = nil;
    }
    if (video_session != nil) {
      [video_session release];
      video_session = nil;
    }
    if (audio_engine != nil) {
      AVAudioInputNode* input = [audio_engine inputNode];
      [input removeTapOnBus:0];
      [audio_engine stop];
      [audio_engine release];
      audio_engine = nil;
    }
    if (audio_converter != nil) {
      [audio_converter release];
      audio_converter = nil;
    }
    if (target_audio_format != nil) {
      [target_audio_format release];
      target_audio_format = nil;
    }
    audio_accumulator.clear();
    const bool was_capturing = capturing;
    capturing = false;
    if (was_capturing && config.on_capture_changed) {
      config.on_capture_changed(false);
    }
  }

  bool StartVideoLocked() {
    video_session = [[AVCaptureSession alloc] init];
    if (video_session == nil) {
      ReportError("Failed to create camera capture session.");
      return false;
    }
    [video_session beginConfiguration];
    if ([video_session canSetSessionPreset:AVCaptureSessionPreset640x480]) {
      [video_session setSessionPreset:AVCaptureSessionPreset640x480];
    }

    AVCaptureDevice* camera
        = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if (camera == nil) {
      [video_session commitConfiguration];
      ReportError("No camera device is available.");
      return false;
    }
    NSError* device_error = nil;
    AVCaptureDeviceInput* input
        = [AVCaptureDeviceInput deviceInputWithDevice:camera
                                                error:&device_error];
    if (input == nil || device_error != nil) {
      [video_session commitConfiguration];
      ReportError(NSErrorMessage(device_error, "Failed to open camera device."));
      return false;
    }
    if (![video_session canAddInput:input]) {
      [video_session commitConfiguration];
      ReportError("Camera input cannot be added to capture session.");
      return false;
    }
    [video_session addInput:input];

    video_output = [[AVCaptureVideoDataOutput alloc] init];
    video_output.alwaysDiscardsLateVideoFrames = YES;
    video_output.videoSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
    };
    if (![video_session canAddOutput:video_output]) {
      [video_session commitConfiguration];
      ReportError("Video output cannot be added to capture session.");
      return false;
    }
    [video_session addOutput:video_output];

    video_delegate = [[CortextWebcamVideoDelegate alloc] initWithOwner:this];
    video_queue = dispatch_queue_create("ai.augmem.cortext.chat.webcam.video",
                                        DISPATCH_QUEUE_SERIAL);
    [video_output setSampleBufferDelegate:video_delegate queue:video_queue];

    [video_session commitConfiguration];
    [video_session startRunning];
    return true;
  }

  bool StartAudioLocked() {
    audio_engine = [[AVAudioEngine alloc] init];
    AVAudioInputNode* input = [audio_engine inputNode];
    if (input == nil) {
      ReportError("No microphone device is available.");
      return false;
    }

    AVAudioFormat* input_format = [input inputFormatForBus:0];
    target_audio_format = [[AVAudioFormat alloc]
        initWithCommonFormat:AVAudioPCMFormatFloat32
                  sampleRate:config.audio_sample_rate
                    channels:1
                 interleaved:NO];
    audio_converter = [[AVAudioConverter alloc] initFromFormat:input_format
                                                      toFormat:target_audio_format];
    if (audio_converter == nil) {
      ReportError("Failed to create direct microphone converter.");
      return false;
    }

    __block WebcamSession::Impl* weak_self = this;
    [input installTapOnBus:0
                bufferSize:1024
                    format:input_format
                     block:^(AVAudioPCMBuffer* buffer, AVAudioTime* when) {
      (void)when;
      if (weak_self != nullptr) {
        weak_self->HandleAudioBuffer(buffer, input_format);
      }
    }];

    NSError* start_error = nil;
    if (![audio_engine startAndReturnError:&start_error]) {
      [input removeTapOnBus:0];
      ReportError(NSErrorMessage(start_error, "Failed to start microphone capture."));
      return false;
    }
    return true;
  }

  void HandleVideoSample(CMSampleBufferRef sample) {
    CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sample);
    if (image_buffer == nullptr) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double min_interval = config.video_fps > 0.0
                                    ? 1.0 / config.video_fps
                                    : 1.0;
    {
      std::lock_guard<std::mutex> lock(video_mu);
      const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                                 now - last_video_emit)
                                 .count();
      if (video_frames_emitted > 0 && elapsed < min_interval) {
        return;
      }
      last_video_emit = now;
      ++video_frames_emitted;
    }

    CVPixelBufferLockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
    const std::size_t width = CVPixelBufferGetWidth(image_buffer);
    const std::size_t height = CVPixelBufferGetHeight(image_buffer);
    const std::size_t stride = CVPixelBufferGetBytesPerRow(image_buffer);
    const auto* base = static_cast<const std::uint8_t*>(
        CVPixelBufferGetBaseAddress(image_buffer));
    if (base == nullptr || width == 0 || height == 0) {
      CVPixelBufferUnlockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
      return;
    }

    WebcamFrame frame;
    frame.width = static_cast<int>(width);
    frame.height = static_cast<int>(height);
    frame.channels = 3;
    frame.rgb.resize(width * height * 3);
    for (std::size_t y = 0; y < height; ++y) {
      const auto* row = base + y * stride;
      for (std::size_t x = 0; x < width; ++x) {
        const std::size_t src = x * 4;
        const std::size_t dst = (y * width + x) * 3;
        frame.rgb[dst + 0] = row[src + 2];
        frame.rgb[dst + 1] = row[src + 1];
        frame.rgb[dst + 2] = row[src + 0];
      }
    }
    CVPixelBufferUnlockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);

    if (config.on_video_frame) {
      config.on_video_frame(std::move(frame));
    }
  }

  void HandleAudioBuffer(AVAudioPCMBuffer* buffer, AVAudioFormat* input_format) {
    if (buffer == nil || input_format == nil || audio_converter == nil
        || target_audio_format == nil) {
      return;
    }

    const double ratio = static_cast<double>(config.audio_sample_rate)
                         / std::max(1.0, input_format.sampleRate);
    const AVAudioFrameCount target_frames
        = static_cast<AVAudioFrameCount>(std::ceil(buffer.frameLength * ratio)) + 16;
    AVAudioPCMBuffer* converted = [[[AVAudioPCMBuffer alloc]
        initWithPCMFormat:target_audio_format
            frameCapacity:target_frames] autorelease];
    if (converted == nil) {
      return;
    }

    __block bool consumed = false;
    NSError* conversion_error = nil;
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
      return buffer;
    };

    const AVAudioConverterOutputStatus status
        = [audio_converter convertToBuffer:converted
                                      error:&conversion_error
                         withInputFromBlock:input_block];
    if (status == AVAudioConverterOutputStatus_Error || conversion_error != nil) {
      ReportError(NSErrorMessage(conversion_error,
                                 "Direct microphone conversion failed."));
      return;
    }
    if (converted.frameLength == 0 || converted.floatChannelData == nil) {
      return;
    }

    const std::size_t chunk_samples = static_cast<std::size_t>(
        std::max(1, config.audio_sample_rate * config.audio_chunk_ms / 1000));
    std::vector<float> ready;
    {
      std::lock_guard<std::mutex> lock(audio_mu);
      const float* data = converted.floatChannelData[0];
      audio_accumulator.insert(audio_accumulator.end(), data,
                               data + converted.frameLength);
      if (audio_accumulator.size() >= chunk_samples) {
        ready.assign(audio_accumulator.begin(),
                     audio_accumulator.begin()
                         + static_cast<std::ptrdiff_t>(chunk_samples));
        audio_accumulator.erase(
            audio_accumulator.begin(),
            audio_accumulator.begin()
                + static_cast<std::ptrdiff_t>(chunk_samples));
      }
    }

    if (!ready.empty() && config.on_audio_chunk) {
      config.on_audio_chunk(std::move(ready));
    }
  }

  void ReportError(const std::string& error) {
    if (config.on_error) {
      config.on_error(error);
    }
  }

  WebcamSessionConfig config;
  bool available = false;
  bool capturing = false;
  std::mutex mu;

  AVCaptureSession* video_session = nil;
  AVCaptureVideoDataOutput* video_output = nil;
  CortextWebcamVideoDelegate* video_delegate = nil;
  dispatch_queue_t video_queue = nullptr;

  AVAudioEngine* audio_engine = nil;
  AVAudioConverter* audio_converter = nil;
  AVAudioFormat* target_audio_format = nil;
  std::mutex audio_mu;
  std::vector<float> audio_accumulator;

  std::mutex video_mu;
  std::chrono::steady_clock::time_point last_video_emit{};
  std::uint64_t video_frames_emitted = 0;
};

struct WebcamSessionVideoBridge {
  static void HandleSample(void* owner, CMSampleBufferRef sample) {
    if (owner != nullptr) {
      static_cast<WebcamSession::Impl*>(owner)->HandleVideoSample(sample);
    }
  }
};

}  // namespace chat

@implementation CortextWebcamVideoDelegate
- (instancetype)initWithOwner:(void*)owner {
  self = [super init];
  if (self != nil) {
    owner_ = owner;
  }
  return self;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  (void)output;
  (void)connection;
  chat::WebcamSessionVideoBridge::HandleSample(owner_, sampleBuffer);
}
@end

namespace chat {

WebcamSession::WebcamSession(WebcamSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

WebcamSession::~WebcamSession() = default;

bool WebcamSession::IsSupported() const {
  return true;
}

bool WebcamSession::IsAvailable() const {
  return impl_ && impl_->available;
}

bool WebcamSession::Start() {
  return impl_ && impl_->Start();
}

void WebcamSession::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

}  // namespace chat
