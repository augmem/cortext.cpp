package cortext

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../build/ffi-release -L${SRCDIR}/../../zig-out/lib -lcortext -Wl,-rpath,${SRCDIR}/../../build/ffi-release -Wl,-rpath,${SRCDIR}/../../zig-out/lib
#cgo linux LDFLAGS: -L${SRCDIR}/../../build/ffi-release -L${SRCDIR}/../../zig-out/lib -lcortext -Wl,-rpath,${SRCDIR}/../../build/ffi-release -Wl,-rpath,${SRCDIR}/../../zig-out/lib
#include <stdlib.h>
#include "cortext/capi.h"
*/
import "C"

import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// Config overrides native defaults. Nil fields retain their native defaults.
type Config struct {
	Focus                  *float64
	Sensitivity            *float64
	Stability              *float64
	AffectInterrupt        *bool
	AffectRetrieval        *bool
	ReinforcementEnabled   *bool
	ProceduralEnabled      *bool
	SequentialEdgesEnabled *bool
	SignalFilterAudio      *bool
	SignalFilterImage      *bool
	SignalFilterText       *bool
}

// Ptr returns a pointer to value for optional Config fields.
func Ptr[T any](value T) *T { return &value }

type Handle struct {
	ptr C.cortext_handle
}

type EmbeddingResult struct {
	Embedding []float32 `json:"embedding"`
	Dimension int       `json:"dimension"`
}

type Media struct {
	Data     []byte
	MimeType string
}

// Retention controls whether an input is naturally gated, committed, bounded,
// or used only for retrieval. Its zero value is Natural.
type Retention int

const (
	RetentionNatural Retention = iota
	RetentionDurable
	RetentionBoundary
	RetentionEphemeral
)

type ProcessOptions struct {
	OmitEmbedding bool
	// Retention is Natural when zero-value / omitted.
	Retention Retention
}

func Version() string {
	return C.GoString(C.cortext_version())
}

func LastError() string {
	return C.GoString(C.cortext_last_error())
}

func New(dbPath string, cfg *Config) (*Handle, error) {
	cDBPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cDBPath))

	var nativeCfg C.cortext_config
	C.cortext_config_init(&nativeCfg)

	if cfg != nil {
		if cfg.Focus != nil {
			nativeCfg.focus = C.double(*cfg.Focus)
		}
		if cfg.Sensitivity != nil {
			nativeCfg.sensitivity = C.double(*cfg.Sensitivity)
		}
		if cfg.Stability != nil {
			nativeCfg.stability = C.double(*cfg.Stability)
		}
		if cfg.AffectInterrupt != nil {
			nativeCfg.affect_interrupt = boolToCInt(*cfg.AffectInterrupt)
		}
		if cfg.AffectRetrieval != nil {
			nativeCfg.affect_retrieval = boolToCInt(*cfg.AffectRetrieval)
		}
		if cfg.ReinforcementEnabled != nil {
			nativeCfg.reinforcement_enabled = boolToCInt(*cfg.ReinforcementEnabled)
		}
		if cfg.ProceduralEnabled != nil {
			nativeCfg.procedural_enabled = boolToCInt(*cfg.ProceduralEnabled)
		}
		if cfg.SequentialEdgesEnabled != nil {
			nativeCfg.sequential_edges_enabled = boolToCInt(*cfg.SequentialEdgesEnabled)
		}
		if cfg.SignalFilterAudio != nil {
			nativeCfg.signal_filter_audio_enabled = boolToCInt(*cfg.SignalFilterAudio)
		}
		if cfg.SignalFilterImage != nil {
			nativeCfg.signal_filter_image_enabled = boolToCInt(*cfg.SignalFilterImage)
		}
		if cfg.SignalFilterText != nil {
			nativeCfg.signal_filter_text_enabled = boolToCInt(*cfg.SignalFilterText)
		}
	}

	handle := C.cortext_create_with_config(&nativeCfg, cDBPath)
	if handle == nil {
		return nil, lastError()
	}

	result := &Handle{ptr: handle}
	runtime.SetFinalizer(result, func(h *Handle) {
		h.Close()
	})
	return result, nil
}

func (h *Handle) Close() {
	if h == nil || h.ptr == nil {
		return
	}
	C.cortext_free(h.ptr)
	h.ptr = nil
}

func (h *Handle) ProcessTextJSON(text string, sourceID string) ([]byte, error) {
	return h.ProcessTextJSONWithOptions(text, sourceID, nil)
}

func (h *Handle) ProcessTextJSONWithOptions(text string, sourceID string, options *ProcessOptions) ([]byte, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	nativeOptions := makeNativeProcessOptions(options)
	raw := C.cortext_process_text_json_with_options(
		h.ptr,
		cText,
		cSourceID,
		&nativeOptions,
	)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) ProcessText(text string, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessTextJSON(text, sourceID))
}

func (h *Handle) ProcessTextWithOptions(text string, sourceID string, options *ProcessOptions) (map[string]any, error) {
	return decodeJSONObject(h.ProcessTextJSONWithOptions(text, sourceID, options))
}

func (h *Handle) EmbedTextJSON(text string) ([]byte, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	raw := C.cortext_embed_text_json(h.ptr, cText)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) EmbedText(text string) ([]float32, error) {
	result, err := decodeEmbeddingResult(h.EmbedTextJSON(text))
	if err != nil {
		return nil, err
	}
	return result.Embedding, nil
}

func (h *Handle) ProcessAudioJSON(pcm []float32, sourceID string) ([]byte, error) {
	return h.ProcessAudioJSONWithOptions(pcm, sourceID, nil)
}

func (h *Handle) ProcessAudioJSONWithOptions(pcm []float32, sourceID string, options *ProcessOptions) ([]byte, error) {
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawPCM *C.float
	if len(pcm) > 0 {
		rawPCM = (*C.float)(unsafe.Pointer(unsafe.SliceData(pcm)))
	}

	nativeOptions := makeNativeProcessOptions(options)
	raw := C.cortext_process_audio_json_with_options(
		h.ptr,
		rawPCM,
		C.size_t(len(pcm)),
		cSourceID,
		&nativeOptions,
	)
	runtime.KeepAlive(pcm)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) ProcessAudioWithMediaJSON(pcm []float32, sourceID string, media *Media) ([]byte, error) {
	return h.ProcessAudioWithMediaJSONWithOptions(pcm, sourceID, media, nil)
}

func (h *Handle) ProcessAudioWithMediaJSONWithOptions(pcm []float32, sourceID string, media *Media, options *ProcessOptions) ([]byte, error) {
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawPCM *C.float
	if len(pcm) > 0 {
		rawPCM = (*C.float)(unsafe.Pointer(unsafe.SliceData(pcm)))
	}

	nativeMedia, mediaData, mediaMime := makeNativeMedia(media)
	defer freeNativeMedia(mediaData, mediaMime)
	nativeOptions := makeNativeProcessOptions(options)

	raw := C.cortext_process_audio_with_media_json_with_options(
		h.ptr,
		rawPCM,
		C.size_t(len(pcm)),
		cSourceID,
		&nativeMedia,
		&nativeOptions,
	)
	runtime.KeepAlive(pcm)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) ProcessAudio(pcm []float32, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessAudioJSON(pcm, sourceID))
}

func (h *Handle) ProcessAudioWithOptions(pcm []float32, sourceID string, options *ProcessOptions) (map[string]any, error) {
	return decodeJSONObject(h.ProcessAudioJSONWithOptions(pcm, sourceID, options))
}

func (h *Handle) ProcessAudioWithMedia(pcm []float32, sourceID string, media *Media) (map[string]any, error) {
	return decodeJSONObject(h.ProcessAudioWithMediaJSON(pcm, sourceID, media))
}

func (h *Handle) ProcessAudioWithMediaAndOptions(pcm []float32, sourceID string, media *Media, options *ProcessOptions) (map[string]any, error) {
	return decodeJSONObject(h.ProcessAudioWithMediaJSONWithOptions(pcm, sourceID, media, options))
}

func (h *Handle) EmbedAudioJSON(pcm []float32) ([]byte, error) {
	var rawPCM *C.float
	if len(pcm) > 0 {
		rawPCM = (*C.float)(unsafe.Pointer(unsafe.SliceData(pcm)))
	}

	raw := C.cortext_embed_audio_json(h.ptr, rawPCM, C.size_t(len(pcm)))
	runtime.KeepAlive(pcm)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) EmbedAudio(pcm []float32) ([]float32, error) {
	result, err := decodeEmbeddingResult(h.EmbedAudioJSON(pcm))
	if err != nil {
		return nil, err
	}
	return result.Embedding, nil
}

func (h *Handle) ProcessImageJSON(data []byte, width int, height int, channels int, sourceID string) ([]byte, error) {
	return h.ProcessImageJSONWithOptions(data, width, height, channels, sourceID, nil)
}

func (h *Handle) ProcessImageJSONWithOptions(data []byte, width int, height int, channels int, sourceID string, options *ProcessOptions) ([]byte, error) {
	if err := validateImageBuffer(data, width, height, channels); err != nil {
		return nil, err
	}
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawData *C.uint8_t
	if len(data) > 0 {
		rawData = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(data)))
	}

	nativeOptions := makeNativeProcessOptions(options)
	raw := C.cortext_process_image_json_with_options(
		h.ptr,
		rawData,
		C.int(width),
		C.int(height),
		C.int(channels),
		cSourceID,
		&nativeOptions,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) ProcessImageWithMediaJSON(data []byte, width int, height int, channels int, sourceID string, media *Media) ([]byte, error) {
	return h.ProcessImageWithMediaJSONWithOptions(data, width, height, channels, sourceID, media, nil)
}

func (h *Handle) ProcessImageWithMediaJSONWithOptions(data []byte, width int, height int, channels int, sourceID string, media *Media, options *ProcessOptions) ([]byte, error) {
	if err := validateImageBuffer(data, width, height, channels); err != nil {
		return nil, err
	}
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawData *C.uint8_t
	if len(data) > 0 {
		rawData = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(data)))
	}

	nativeMedia, mediaData, mediaMime := makeNativeMedia(media)
	defer freeNativeMedia(mediaData, mediaMime)
	nativeOptions := makeNativeProcessOptions(options)

	raw := C.cortext_process_image_with_media_json_with_options(
		h.ptr,
		rawData,
		C.int(width),
		C.int(height),
		C.int(channels),
		cSourceID,
		&nativeMedia,
		&nativeOptions,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) ProcessImage(data []byte, width int, height int, channels int, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessImageJSON(data, width, height, channels, sourceID))
}

func (h *Handle) ProcessImageWithOptions(data []byte, width int, height int, channels int, sourceID string, options *ProcessOptions) (map[string]any, error) {
	return decodeJSONObject(h.ProcessImageJSONWithOptions(data, width, height, channels, sourceID, options))
}

func (h *Handle) ProcessImageWithMedia(data []byte, width int, height int, channels int, sourceID string, media *Media) (map[string]any, error) {
	return decodeJSONObject(h.ProcessImageWithMediaJSON(data, width, height, channels, sourceID, media))
}

func (h *Handle) ProcessImageWithMediaAndOptions(data []byte, width int, height int, channels int, sourceID string, media *Media, options *ProcessOptions) (map[string]any, error) {
	return decodeJSONObject(h.ProcessImageWithMediaJSONWithOptions(data, width, height, channels, sourceID, media, options))
}

func (h *Handle) EmbedImageJSON(data []byte, width int, height int, channels int) ([]byte, error) {
	if err := validateImageBuffer(data, width, height, channels); err != nil {
		return nil, err
	}
	var rawData *C.uint8_t
	if len(data) > 0 {
		rawData = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(data)))
	}

	raw := C.cortext_embed_image_json(
		h.ptr,
		rawData,
		C.int(width),
		C.int(height),
		C.int(channels),
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) EmbedImage(data []byte, width int, height int, channels int) ([]float32, error) {
	result, err := decodeEmbeddingResult(h.EmbedImageJSON(data, width, height, channels))
	if err != nil {
		return nil, err
	}
	return result.Embedding, nil
}

func (h *Handle) ConsolidateJSON() ([]byte, error) {
	raw := C.cortext_consolidate_json(h.ptr)
	runtime.KeepAlive(h)
	return takeJSONString(raw)
}

func (h *Handle) Consolidate() (map[string]any, error) {
	return decodeJSONObject(h.ConsolidateJSON())
}

func (h *Handle) Flush() error {
	status := C.cortext_flush(h.ptr)
	runtime.KeepAlive(h)
	if status != 0 {
		return lastError()
	}
	return nil
}

func (h *Handle) Reset() error {
	status := C.cortext_reset(h.ptr)
	runtime.KeepAlive(h)
	if status != 0 {
		return lastError()
	}
	return nil
}

func boolToCInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func validateImageBuffer(data []byte, width int, height int, channels int) error {
	if width <= 0 || height <= 0 || channels <= 0 {
		return errors.New("width, height, and channels must be positive")
	}
	const maxCInt = int64(1<<31 - 1)
	if int64(width) > maxCInt || int64(height) > maxCInt || int64(channels) > maxCInt {
		return errors.New("image dimensions exceed the native int range")
	}
	maxInt := int(^uint(0) >> 1)
	if width > maxInt/height || width*height > maxInt/channels {
		return errors.New("image dimensions overflow addressable memory")
	}
	expected := width * height * channels
	if len(data) < expected {
		return errors.New("image buffer is smaller than width * height * channels")
	}
	return nil
}

func makeNativeMedia(media *Media) (C.cortext_media, unsafe.Pointer, *C.char) {
	if media == nil || len(media.Data) == 0 {
		return C.cortext_media{}, nil, nil
	}

	mediaData := C.CBytes(media.Data)
	mediaMime := C.CString(media.MimeType)
	native := C.cortext_media{
		data:     (*C.uint8_t)(mediaData),
		size:     C.size_t(len(media.Data)),
		mimetype: mediaMime,
	}
	return native, mediaData, mediaMime
}

func makeNativeProcessOptions(options *ProcessOptions) C.cortext_process_json_options {
	native := C.cortext_process_json_options{}
	native.struct_size = C.size_t(unsafe.Sizeof(native))
	C.cortext_process_json_options_init(&native)
	if options != nil {
		if options.OmitEmbedding {
			native.include_embedding = 0
		}
		if options.Retention != 0 {
			native.retention = nativeRetention(options.Retention)
		}
	}
	return native
}

func nativeRetention(retention Retention) C.int {
	switch retention {
	case RetentionDurable:
		return C.CORTEXT_RETENTION_DURABLE
	case RetentionBoundary:
		return C.CORTEXT_RETENTION_BOUNDARY
	case RetentionEphemeral:
		return C.CORTEXT_RETENTION_EPHEMERAL
	default:
		return C.CORTEXT_RETENTION_NATURAL
	}
}

func freeNativeMedia(mediaData unsafe.Pointer, mediaMime *C.char) {
	if mediaData != nil {
		C.free(mediaData)
	}
	if mediaMime != nil {
		C.free(unsafe.Pointer(mediaMime))
	}
}

func lastError() error {
	msg := LastError()
	if msg == "" {
		return errors.New("cortext call failed")
	}
	return errors.New(msg)
}

func takeJSONString(raw *C.char) ([]byte, error) {
	if raw == nil {
		return nil, lastError()
	}
	defer C.cortext_string_free(raw)
	return []byte(C.GoString(raw)), nil
}

func decodeJSONObject(payload []byte, err error) (map[string]any, error) {
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if unmarshalErr := json.Unmarshal(payload, &out); unmarshalErr != nil {
		return nil, unmarshalErr
	}
	return out, nil
}

func decodeEmbeddingResult(payload []byte, err error) (EmbeddingResult, error) {
	if err != nil {
		return EmbeddingResult{}, err
	}
	var out EmbeddingResult
	if unmarshalErr := json.Unmarshal(payload, &out); unmarshalErr != nil {
		return EmbeddingResult{}, unmarshalErr
	}
	return out, nil
}
