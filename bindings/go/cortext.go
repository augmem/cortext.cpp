package cortext

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../zig-out/lib -L${SRCDIR}/../../build/ffi-release -lcortext -Wl,-rpath,${SRCDIR}/../../zig-out/lib -Wl,-rpath,${SRCDIR}/../../build/ffi-release
#cgo linux LDFLAGS: -L${SRCDIR}/../../zig-out/lib -L${SRCDIR}/../../build/ffi-release -lcortext -Wl,-rpath,${SRCDIR}/../../zig-out/lib -Wl,-rpath,${SRCDIR}/../../build/ffi-release
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

type ConsolidationMode int

const (
	ConsolidateShallow ConsolidationMode = ConsolidateShallowValue
	ConsolidateDeep    ConsolidationMode = ConsolidateDeepValue
	ConsolidateBoth    ConsolidationMode = ConsolidateBothValue
)

const (
	ConsolidateShallowValue ConsolidationMode = 0
	ConsolidateDeepValue    ConsolidationMode = 1
	ConsolidateBothValue    ConsolidationMode = 2
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
	LabelBankPath          string
	// SummarizerProviderURI routes the Summarizer role to an external
	// inference provider (e.g. "ollama://127.0.0.1:11435/gemma4:e2b").
	// Empty keeps local model auto-discovery; a URI that cannot be
	// resolved and verified makes New fail.
	SummarizerProviderURI string
	// ExtractorProviderURI routes the Extractor role; same semantics as
	// SummarizerProviderURI.
	ExtractorProviderURI string
}

type Handle struct {
	ptr C.cortext_handle
}

type EmbeddingResult struct {
	Embedding []float32 `json:"embedding"`
	Dimension int       `json:"dimension"`
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

	var labelBankCleanup func()
	var summarizerURICleanup func()
	var extractorURICleanup func()
	if cfg != nil {
		nativeCfg.focus = C.double(cfg.Focus)
		nativeCfg.sensitivity = C.double(cfg.Sensitivity)
		nativeCfg.stability = C.double(cfg.Stability)
		nativeCfg.affect_interrupt = boolToCInt(cfg.AffectInterrupt)
		nativeCfg.affect_retrieval = boolToCInt(cfg.AffectRetrieval)
		nativeCfg.reinforcement_enabled = boolToCInt(cfg.ReinforcementEnabled)
		nativeCfg.procedural_enabled = boolToCInt(cfg.ProceduralEnabled)
		nativeCfg.sequential_edges_enabled = boolToCInt(cfg.SequentialEdgesEnabled)
		if cfg.LabelBankPath != "" {
			labelBankPath := C.CString(cfg.LabelBankPath)
			nativeCfg.label_bank_path = labelBankPath
			labelBankCleanup = func() {
				C.free(unsafe.Pointer(labelBankPath))
			}
		}
		if cfg.SummarizerProviderURI != "" {
			summarizerURI := C.CString(cfg.SummarizerProviderURI)
			nativeCfg.summarizer_provider_uri = summarizerURI
			summarizerURICleanup = func() {
				C.free(unsafe.Pointer(summarizerURI))
			}
		}
		if cfg.ExtractorProviderURI != "" {
			extractorURI := C.CString(cfg.ExtractorProviderURI)
			nativeCfg.extractor_provider_uri = extractorURI
			extractorURICleanup = func() {
				C.free(unsafe.Pointer(extractorURI))
			}
		}
	}
	if labelBankCleanup != nil {
		defer labelBankCleanup()
	}
	if summarizerURICleanup != nil {
		defer summarizerURICleanup()
	}
	if extractorURICleanup != nil {
		defer extractorURICleanup()
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
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	raw := C.cortext_process_text_json(h.ptr, cText, cSourceID)
	return takeJSONString(raw)
}

func (h *Handle) ProcessText(text string, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessTextJSON(text, sourceID))
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
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawPCM *C.float
	if len(pcm) > 0 {
		rawPCM = (*C.float)(unsafe.Pointer(unsafe.SliceData(pcm)))
	}

	raw := C.cortext_process_audio_json(h.ptr, rawPCM, C.size_t(len(pcm)), cSourceID)
	return takeJSONString(raw)
}

func (h *Handle) ProcessAudio(pcm []float32, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessAudioJSON(pcm, sourceID))
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
	cSourceID := C.CString(sourceID)
	defer C.free(unsafe.Pointer(cSourceID))

	var rawData *C.uint8_t
	if len(data) > 0 {
		rawData = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(data)))
	}

	raw := C.cortext_process_image_json(
		h.ptr,
		rawData,
		C.int(width),
		C.int(height),
		C.int(channels),
		cSourceID,
	)
	return takeJSONString(raw)
}

func (h *Handle) ProcessImage(data []byte, width int, height int, channels int, sourceID string) (map[string]any, error) {
	return decodeJSONObject(h.ProcessImageJSON(data, width, height, channels, sourceID))
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

func (h *Handle) ConsolidateModeJSON(mode ConsolidationMode) ([]byte, error) {
	raw := C.cortext_consolidate_mode_json(h.ptr, C.int(mode))
	return takeJSONString(raw)
}

func (h *Handle) ConsolidateMode(mode ConsolidationMode) (map[string]any, error) {
	return decodeJSONObject(h.ConsolidateModeJSON(mode))
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
