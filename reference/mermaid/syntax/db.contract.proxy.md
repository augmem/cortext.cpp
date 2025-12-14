# Database Contract - Proxy Module

```mermaid
erDiagram
    "proxy.routes" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) name
        TEXT description
        route_kind route_kind
        VARCHAR(255) host_pattern
        VARCHAR(500) path_pattern
        VARCHAR(50) method_pattern
        JSONB header_rules
        VARCHAR(1000) target_url
        JSONB[] target_servers
        load_balance_method load_balance_method
        JSONB add_headers
        TEXT[] remove_headers
        VARCHAR(500) rewrite_path
        JSONB add_response_headers
        TEXT[] remove_response_headers
        TEXT[] allowed_origins
        TEXT[] allowed_methods
        BOOLEAN require_auth
        VARCHAR(50) auth_type
        INTEGER rate_limit_requests
        INTEGER rate_limit_window_seconds
        BOOLEAN circuit_breaker_enabled
        INTEGER circuit_breaker_threshold
        INTEGER circuit_breaker_timeout_seconds
        INTEGER connect_timeout_ms
        INTEGER read_timeout_ms
        INTEGER write_timeout_ms
        BOOLEAN retry_enabled
        INTEGER retry_max_attempts
        INTEGER retry_backoff_ms
        BOOLEAN cache_enabled
        INTEGER cache_ttl_seconds
        VARCHAR(500) cache_key_pattern
        BOOLEAN is_active
        INTEGER priority
        BIGINT total_requests
        BIGINT total_errors
        INTEGER avg_response_time_ms
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        UUID created_by FK
        VARCHAR(50) owner_slug
        VARCHAR(50) workspace_slug
        VARCHAR(50) server_slug
        UUID tool_server_id FK
    }
    "proxy.route_metrics" {
        UUID id PK
        UUID route_id FK
        TIMESTAMPTZ time_bucket
        INTEGER request_count
        INTEGER error_count
        INTEGER status_2xx
        INTEGER status_3xx
        INTEGER status_4xx
        INTEGER status_5xx
        INTEGER avg_response_time_ms
        INTEGER p50_response_time_ms
        INTEGER p95_response_time_ms
        INTEGER p99_response_time_ms
        INTEGER max_response_time_ms
        BIGINT bytes_in
        BIGINT bytes_out
        INTEGER cache_hits
        INTEGER cache_misses
        INTEGER circuit_breaker_opens
        TIMESTAMPTZ created_at
    }
    "proxy.health_checks" {
        UUID id PK
        UUID route_id FK
        BOOLEAN enabled
        INTEGER interval_seconds
        INTEGER timeout_seconds
        VARCHAR(500) check_path
        VARCHAR(10) check_method
        INTEGER[] expected_status
        TEXT expected_body_contains
        INTEGER healthy_threshold
        INTEGER unhealthy_threshold
        BOOLEAN is_healthy
        INTEGER consecutive_successes
        INTEGER consecutive_failures
        TIMESTAMPTZ last_check_at
        TIMESTAMPTZ last_healthy_at
        TEXT last_error
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "proxy.api_keys" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) name
        TEXT description
        VARCHAR(255) key_hash
        VARCHAR(10) key_prefix
        api_key_role role
        JSONB permissions
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ last_used_at
        INET last_used_ip
        TIMESTAMPTZ revoked_at
        UUID revoked_by FK
        TEXT revoke_reason
        INTEGER rate_limit_requests
        INTEGER rate_limit_window_seconds
        BIGINT total_requests
        BIGINT total_errors
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        UUID created_by FK
    }
    "proxy.server_auth_configs" {
        UUID id PK
        UUID route_id FK
        auth_method auth_method
        BYTEA encrypted_credentials
        BYTEA encryption_iv
        VARCHAR(50) injection_type
        VARCHAR(255) injection_key
        VARCHAR(50) injection_prefix
        JSONB custom_headers
        BOOLEAN is_active
        VARCHAR(500) test_endpoint
        TIMESTAMPTZ last_test_at
        BOOLEAN last_test_success
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        UUID created_by FK
    }
    "proxy.api_key_audit_log" {
        UUID id PK
        UUID api_key_id FK
        TIMESTAMPTZ timestamp
        VARCHAR(10) method
        VARCHAR(1000) path
        INTEGER status_code
        INTEGER response_time_ms
        INET ip_address
        TEXT user_agent
        UUID route_id FK
        TEXT error_message
        BIGINT bytes_in
        BIGINT bytes_out
    }

    "proxy.routes" }o--|| "platform.workspaces" : "workspace_id"
    "proxy.routes" }o--|| "auth.users" : "created_by"
    "proxy.routes" }o--|| "agent.tool_servers" : "tool_server_id"
    "proxy.route_metrics" }o--|| "proxy.routes" : "route_id"
    "proxy.health_checks" }o--|| "proxy.routes" : "route_id"
    "proxy.api_keys" }o--|| "platform.workspaces" : "workspace_id"
    "proxy.api_keys" }o--|| "auth.users" : "revoked_by"
    "proxy.api_keys" }o--|| "auth.users" : "created_by"
    "proxy.server_auth_configs" }o--|| "proxy.routes" : "route_id"
    "proxy.server_auth_configs" }o--|| "auth.users" : "created_by"
    "proxy.api_key_audit_log" }o--|| "proxy.api_keys" : "api_key_id"
    "proxy.api_key_audit_log" }o--|| "proxy.routes" : "route_id"
```</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/db.contract.proxy.md
