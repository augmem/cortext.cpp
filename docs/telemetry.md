# cortext telemetry (OpenTelemetry)

`cortext` can emit **traces, metrics, and logs** via the OpenTelemetry C++ SDK
([`opentelemetry-cpp`](https://github.com/open-telemetry/opentelemetry-cpp)).

## Build-time enablement

On native builds, `cortext` always compiles with the OpenTelemetry **API** and
emits spans/metrics/logs to the **global providers**. If the application does
not install an OpenTelemetry SDK provider/exporter pipeline, instrumentation is
effectively **noop**.

On WebAssembly (Emscripten) builds, telemetry is stubbed out.

## Runtime configuration

`cortext` does not configure exporters/providers. The **application** is
responsible for installing SDK providers/exporters (often configured via
standard `OTEL_*` environment variables in the app).

Common environment variables (when the application uses the OTEL SDK):

* `OTEL_SERVICE_NAME`: service name (defaults to `cortext` if unset)
* `OTEL_RESOURCE_ATTRIBUTES`: resource attributes (e.g. `service.version=...`)
* `OTEL_EXPORTER_OTLP_ENDPOINT`: collector endpoint
* `OTEL_EXPORTER_OTLP_PROTOCOL`: `grpc` or `http/protobuf`
* `OTEL_TRACES_EXPORTER`: `otlp` or `none`
* `OTEL_METRICS_EXPORTER`: `otlp` or `none`
* `OTEL_LOGS_EXPORTER`: `otlp` or `none`
* `OTEL_TRACES_SAMPLER`: e.g. `parentbased_always_on`, `traceidratio`, `always_off`
* `OTEL_TRACES_SAMPLER_ARG`: ratio for `traceidratio` (e.g. `0.1`)

References:

* OpenTelemetry C++ docs: <https://opentelemetry.io/docs/languages/cpp/library/>
* SDK environment variables spec: <https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/>

## Data model / cardinality policy

### Metrics

Metrics are intentionally **label-free** (no attributes) to avoid cardinality
explosions. If a label is needed in the future, add it only with a clear,
bounded value set.

Baseline metrics include:

* `cortext.process_duration_ms` (histogram)
* `cortext.signals_processed_total` (counter)
* `cortext.at_boundary_total` (counter)
* `cortext.interrupt_allowed_total` (counter)
* `cortext.flush_total` (counter)
* `cortext.episode_commit_total` (counter)

OperationContext-derived behavior metrics include:

* `cortext.metric.*` histograms for Algorithm-7 normalized metrics
* `cortext.threshold_T_dynamic`, `cortext.threshold_hysteresis`
* `cortext.effective_focus`, `cortext.coherence`, `cortext.emotion_intensity`
* `cortext.mni_*` diagnostics
* `cortext.last_weight_sum`, `cortext.last_effective_metric_count`

### Traces

Tracing is designed to propagate through operations:

* `cortext.process` is the root span for a `SignalProcessor::Process()` call.
* Each `IOperation::Execute()` is wrapped by `cortext.operation` with
  `cortext.operation_type`.
* Database spans `cortext.db.execute` attach under the currently active
  operation span and include `db.system=sqlite` and low-cardinality
  `db.operation` (no raw SQL statements).

### Logs

Logs are **error-focused** and include structured low-cardinality fields like
`component`, `db.system`, and `db.operation`. Sensitive/high-cardinality payload
(raw SQL, embeddings, user content) is intentionally not logged.
