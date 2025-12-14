# Database Contract - Agent Module

```mermaid
erDiagram
    "agent.registry" {
        UUID id PK
        VARCHAR(255) name
        VARCHAR(255) display_name
        TEXT description
        VARCHAR(50) version
        VARCHAR(255) author
        TEXT[] supported_protocols
        TEXT[] required_tool_servers
        TEXT[] supported_models
        BOOLEAN is_public
        BOOLEAN is_featured
        INTEGER install_count
        NUMERIC rating_average
        INTEGER rating_count
        JSONB default_config
        UUID schema_id
        TEXT[] tags
        TEXT documentation_url
        TEXT repository_url
        TEXT icon_url
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "agent.sessions" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) external_session_id
        UUID agent_id FK
        VARCHAR(255) agent_identifier
        JSONB agent_metadata
        session_status status
        TIMESTAMPTZ started_at
        TIMESTAMPTZ ended_at
        INTEGER tool_calls_count
        INTEGER total_duration_ms
        INTEGER total_input_tokens
        INTEGER total_output_tokens
        NUMERIC total_cost_usd
        VARCHAR(255) detected_model
        NUMERIC model_confidence
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "agent.tool_servers" {
        UUID id PK
        UUID workspace_id FK
        UUID registry_id
        VARCHAR(255) name
        VARCHAR(255) slug
        VARCHAR(50) protocol
        TEXT endpoint_url
        UUID schema_id
        BOOLEAN is_active
        VARCHAR(50) health_status
        TIMESTAMPTZ last_health_check
        JSONB configuration
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ last_heartbeat
        INTEGER response_time_ms
    }
    "agent.tool_call_log" {
        UUID id PK
        UUID session_id FK
        UUID tool_server_id FK
        VARCHAR(255) tool_name
        VARCHAR(50) protocol
        tool_call_status status
        JSONB request_params
        JSONB response_data
        TEXT error_message
        TIMESTAMPTZ started_at
        TIMESTAMPTZ completed_at
        INTEGER duration_ms
        INTEGER input_tokens
        INTEGER output_tokens
        NUMERIC estimated_cost_usd
        JSONB metadata
        TIMESTAMPTZ created_at
        VARCHAR(64) trace_id
    }
    "agent.error_log" {
        UUID id PK
        UUID agent_id
        UUID session_id
        VARCHAR(100) error_type
        TEXT error_message
        TEXT stack_trace
        JSONB context
        TIMESTAMPTZ occurred_at
    }

    "agent.sessions" }o--|| "platform.workspaces" : "workspace_id"
    "agent.sessions" }o--|| "agent.registry" : "agent_id"
    "agent.tool_servers" }o--|| "platform.workspaces" : "workspace_id"
    "agent.tool_call_log" }o--|| "agent.sessions" : "session_id"
    "agent.tool_call_log" }o--|| "agent.tool_servers" : "tool_server_id"
```</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/db.contract.agent.md has been created.
