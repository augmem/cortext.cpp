package cortext

import "testing"

func TestVersionIsNonEmpty(t *testing.T) {
	if Version() == "" {
		t.Fatal("Version() returned an empty string")
	}
}

func TestValidateImageBuffer(t *testing.T) {
	if err := validateImageBuffer(make([]byte, 12), 2, 2, 3); err != nil {
		t.Fatalf("valid image rejected: %v", err)
	}
	if err := validateImageBuffer(make([]byte, 10), 1000, 1000, 3); err == nil {
		t.Fatal("short image buffer was accepted")
	}
	if err := validateImageBuffer(nil, 0, 1, 3); err == nil {
		t.Fatal("non-positive dimensions were accepted")
	}
}

func TestConfigFieldsDistinguishOmittedAndExplicitZero(t *testing.T) {
	cfg := Config{Focus: Ptr(0.0), AffectInterrupt: Ptr(false)}
	if cfg.Focus == nil || *cfg.Focus != 0.0 {
		t.Fatal("explicit zero focus was not preserved")
	}
	if cfg.Sensitivity != nil {
		t.Fatal("omitted sensitivity should remain nil and use the native default")
	}
	if cfg.AffectInterrupt == nil || *cfg.AffectInterrupt {
		t.Fatal("explicit false toggle was not preserved")
	}
}
