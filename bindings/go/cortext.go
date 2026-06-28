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

type Config struct {
	Focus                  float64
	Sensitivity            float64
	Stability              float64
	AffectInterrupt        bool
	AffectRetrieval        bool
	ReinforcementEnabled   bool
	ProceduralEnabled      bool
	SequentialEdgesEnabled bool
	SignalFilterAudio      bool
	SignalFilterImage      bool
	SignalFilterText       bool
}

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

type ProcessOptions struct {
	OmitEmbedding bool
}

func Version() string {
	return C.GoString(C.cortext_version())
}

func LastError() string {
	return C.GoString(C.cortext_last_error())
}

func New(dbPath string, modelsDir string, cfg *Config) (*Handle, error) {
	cDBPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cDBPath))

	var cModelsDir *C.char
	if modelsDir != "" {
		cModelsDir = C.CString(modelsDir)
		defer C.free(unsafe.Pointer(cModelsDir))
	}

	var nativeCfg C.cortext_config
	C.cortext_config_init(&nativeCfg)

	if cfg != nil {
		nativeCfg.focus = C.double(cfg.Focus)
		nativeCfg.sensitivity = C.double(cfg.Sensitivity)
		nativeCfg.stability = C.double(cfg.Stability)
		nativeCfg.affect_interrupt = boolToCInt(cfg.AffectInterrupt)
		nativeCfg.affect_retrieval = boolToCInt(cfg.AffectRetrieval)
		nativeCfg.reinforcement_enabled = boolToCInt(cfg.ReinforcementEnabled)
		nativeCfg.procedural_enabled = boolToCInt(cfg.ProceduralEnabled)
		nativeCfg.sequential_edges_enabled = boolToCInt(cfg.SequentialEdgesEnabled)
		nativeCfg.signal_filter_audio_enabled = boolToCInt(cfg.SignalFilterAudio)
		nativeCfg.signal_filter_image_enabled = boolToCInt(cfg.SignalFilterImage)
		nativeCfg.signal_filter_text_enabled = boolToCInt(cfg.SignalFilterText)
	}

	handle := C.cortext_create_with_config(&nativeCfg, cDBPath, cModelsDir)
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
	return takeJSONString(raw)
}

func (h *Handle) ProcessImageWithMediaJSON(data []byte, width int, height int, channels int, sourceID string, media *Media) ([]byte, error) {
	return h.ProcessImageWithMediaJSONWithOptions(data, width, height, channels, sourceID, media, nil)
}

func (h *Handle) ProcessImageWithMediaJSONWithOptions(data []byte, width int, height int, channels int, sourceID string, media *Media, options *ProcessOptions) ([]byte, error) {
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
	return takeJSONString(raw)
}

func (h *Handle) Consolidate() (map[string]any, error) {
	return decodeJSONObject(h.ConsolidateJSON())
}

func (h *Handle) Flush() error {
	status := C.cortext_flush(h.ptr)
	if status != 0 {
		return lastError()
	}
	return nil
}

func (h *Handle) Reset() error {
	status := C.cortext_reset(h.ptr)
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
	C.cortext_process_json_options_init(&native)
	if options != nil && options.OmitEmbedding {
		native.include_embedding = 0
	}
	return native
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
