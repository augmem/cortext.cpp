package main

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "flag"
    "fmt"
    "os"
    "path/filepath"

    "github.com/tetratelabs/wazero"
    "github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

type fixture struct {
    ID   string   `json:"id"`
    Path string   `json:"path"`
    Tags []string `json:"tags"`
}

type matrix struct {
    Fixtures []fixture `json:"fixtures"`
}

func loadMatrix(bundle string) ([]fixture, error) {
    raw, err := os.ReadFile(filepath.Join(bundle, "matrix.json"))
    if err != nil {
        return nil, err
    }
    var m matrix
    if err := json.Unmarshal(raw, &m); err != nil {
        return nil, err
    }
    if len(m.Fixtures) == 0 {
        return nil, errors.New("matrix has no fixtures")
    }
    return m.Fixtures, nil
}

func runFixture(ctx context.Context, rt wazero.Runtime, compiled wazero.CompiledModule, bundle string, fx fixture) (map[string]any, error) {
    stdoutBuf := bytes.NewBuffer(nil)
    stderrBuf := bytes.NewBuffer(nil)

    fsConfig := wazero.NewFSConfig().WithDirMount(bundle, "/bundle")
    moduleConfig := wazero.NewModuleConfig().
        WithArgs(
            "objstore_wasm_matrix",
            "--bundle-root", "/bundle",
            "--fixture", fx.Path,
            "--fixture-id", fx.ID,
        ).
        WithStdout(stdoutBuf).
        WithStderr(stderrBuf).
        WithFSConfig(fsConfig)

    inst, err := rt.InstantiateModule(ctx, compiled, moduleConfig)
    if inst != nil {
        defer inst.Close(ctx)
    }
    if err != nil {
        return nil, fmt.Errorf("instantiate module: %w", err)
    }

    out := stdoutBuf.String()
    if out == "" {
        return nil, fmt.Errorf("fixture %s produced no output: %s", fx.ID, stderrBuf.String())
    }
    lines := bytes.Split(bytes.TrimSpace(stdoutBuf.Bytes()), []byte("\n"))
    payload := lines[len(lines)-1]
    var result map[string]any
    if err := json.Unmarshal(payload, &result); err != nil {
        return nil, fmt.Errorf("parse json: %w", err)
    }
    return result, nil
}

func main() {
    bundleFlag := flag.String("bundle", "dist/wasm/dev", "path to the packaged bundle")
    moduleFlag := flag.String("module", "", "optional override for the WASI matrix module")
    flag.Parse()

    bundle, err := filepath.Abs(*bundleFlag)
    if err != nil {
        fmt.Fprintf(os.Stderr, "resolve bundle: %v\n", err)
        os.Exit(1)
    }

    modulePath := *moduleFlag
    if modulePath == "" {
        modulePath = filepath.Join(bundle, "wasi", "objstore_wasm_matrix.wasm")
    }

    fixtures, err := loadMatrix(bundle)
    if err != nil {
        fmt.Fprintf(os.Stderr, "load matrix: %v\n", err)
        os.Exit(1)
    }

    ctx := context.Background()
    runtime := wazero.NewRuntime(ctx)
    defer runtime.Close(ctx)

    wasi_snapshot_preview1.MustInstantiate(ctx, runtime)

    wasmBytes, err := os.ReadFile(modulePath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "read module: %v\n", err)
        os.Exit(1)
    }

    compiled, err := runtime.CompileModule(ctx, wasmBytes)
    if err != nil {
        fmt.Fprintf(os.Stderr, "compile module: %v\n", err)
        os.Exit(1)
    }
    defer compiled.Close(ctx)

    var failures []map[string]any

    for _, fx := range fixtures {
        result, runErr := runFixture(ctx, runtime, compiled, bundle, fx)
        status := "ok"
        if runErr != nil {
            status = "failed"
            failures = append(failures, map[string]any{
                "fixture": fx.ID,
                "error":   runErr.Error(),
            })
        } else if resStatus, ok := result["status"].(string); !ok || resStatus != "ok" {
            failures = append(failures, result)
            status = "failed"
        }
        fmt.Printf("[go] %s: %s\n", fx.ID, status)
    }

    if len(failures) > 0 {
        enc := json.NewEncoder(os.Stderr)
        enc.SetIndent("", "  ")
        _ = enc.Encode(failures)
        os.Exit(1)
    }

    fmt.Println("go harness passed all fixtures")
}
