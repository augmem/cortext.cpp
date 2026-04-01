use anyhow::{bail, Context, Result};
use cap_std::{ambient_authority, fs::Dir};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi::pipe::WritePipe;
use wasmtime_wasi::sync::{add_to_linker, WasiCtx, WasiCtxBuilder};

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    path: String,
    #[allow(dead_code)]
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct Matrix {
    fixtures: Vec<Fixture>,
}

fn load_matrix(bundle: &Path) -> Result<Vec<Fixture>> {
    let matrix_path = bundle.join("matrix.json");
    let data = fs::read_to_string(&matrix_path)
        .with_context(|| format!("unable to read {}", matrix_path.display()))?;
    let parsed: Matrix = serde_json::from_str(&data)
        .with_context(|| format!("unable to parse {}", matrix_path.display()))?;
    if parsed.fixtures.is_empty() {
        bail!("matrix has no fixtures: {}", matrix_path.display());
    }
    Ok(parsed.fixtures)
}

fn run_fixture(engine: &Engine, module: &Module, bundle_dir: &Dir, fixture: &Fixture) -> Result<Value> {
    let stdout = WritePipe::new_in_memory();
    let stderr = WritePipe::new_in_memory();
    let stdout_reader = stdout.clone();
    let stderr_reader = stderr.clone();

    let mut args = vec![
        "objstore_wasm_matrix".to_string(),
        "--bundle-root".to_string(),
        "/bundle".to_string(),
        "--fixture".to_string(),
        fixture.path.clone(),
        "--fixture-id".to_string(),
        fixture.id.clone(),
    ];

    let mut builder = WasiCtxBuilder::new();
    builder = builder.args(&args)?;
    builder = builder.stdout(stdout);
    builder = builder.stderr(stderr);
    builder = builder.preopened_dir(bundle_dir.try_clone()?, "/bundle")?;
    let ctx = builder.build();

    let mut store = Store::new(engine, ctx);
    let mut linker = Linker::new(engine);
    add_to_linker(&mut linker, |cx: &mut WasiCtx| cx)?;

    let instance = linker.instantiate(&mut store, module)?;
    let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    start.call(&mut store, ())?;

    let stdout_bytes = stdout_reader
        .try_into_inner()
        .expect("stdout still borrowed")
        .into_inner();
    if stdout_bytes.is_empty() {
        let err_bytes = stderr_reader
            .try_into_inner()
            .expect("stderr still borrowed")
            .into_inner();
        bail!(
            "fixture {} produced no output: {}",
            fixture.id,
            String::from_utf8_lossy(&err_bytes)
        );
    }
    let text = String::from_utf8(stdout_bytes)?;
    let last_line = text.lines().last().unwrap_or("");
    let value: Value = serde_json::from_str(last_line)?;
    Ok(value)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut bundle_path = PathBuf::from("dist/wasm/dev");
    let mut module_override: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bundle" => {
                if let Some(val) = args.next() {
                    bundle_path = PathBuf::from(val);
                }
            }
            "--module" => {
                if let Some(val) = args.next() {
                    module_override = Some(PathBuf::from(val));
                }
            }
            _ => {}
        }
    }

    bundle_path = fs::canonicalize(&bundle_path)
        .with_context(|| format!("unable to resolve {}", bundle_path.display()))?;

    let module_path = module_override
        .map(|p| p)
        .unwrap_or_else(|| bundle_path.join("wasi").join("objstore_wasm_matrix.wasm"));

    let fixtures = load_matrix(&bundle_path)?;
    let bundle_dir = Dir::open_ambient_dir(&bundle_path, ambient_authority())?;

    let engine = Engine::default();
    let module = Module::from_file(&engine, module_path)?;

    let mut failures = Vec::new();

    for fixture in fixtures.iter() {
        match run_fixture(&engine, &module, &bundle_dir, fixture) {
            Ok(value) => {
                let status = value
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                println!("[rust] {}: {}", fixture.id, status);
                if status != "ok" {
                    failures.push(value);
                }
            }
            Err(err) => {
                println!("[rust] {}: failed", fixture.id);
                failures.push(serde_json::json!({
                    "fixture": fixture.id,
                    "error": err.to_string(),
                }));
            }
        }
    }

    if !failures.is_empty() {
        println!("rust harness failures:");
        println!("{}", serde_json::to_string_pretty(&failures)?);
        bail!("rust harness failed");
    }

    println!("rust harness passed all fixtures");
    Ok(())
}
