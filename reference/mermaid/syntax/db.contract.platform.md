# Database Contract - Platform Module

```mermaid
erDiagram
    "platform.accounts" {
        UUID id PK
        UUID user_id FK
        VARCHAR(255) type
        UUID provider_id FK
        VARCHAR(255) provider_account_id
        TEXT refresh_token
        TEXT access_token
        INTEGER expires_at
        VARCHAR(255) token_type
        VARCHAR(255) scope
        TEXT id_token
        VARCHAR(255) session_state
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.workspaces" {
        UUID id PK
        VARCHAR(255) name
        VARCHAR(255) slug
        TEXT description
        UUID owner_id FK
        JSONB settings
        TIMESTAMPTZ deleted_at
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        workspace_type workspace_type
    }
    "platform.signups" {
        UUID id PK
        signup_kind kind
        VARCHAR(255) email
        VARCHAR(20) status
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.logs" {
        UUID id PK
        TIMESTAMPTZ timestamp
        VARCHAR(20) level
        VARCHAR(50) category
        VARCHAR(100) action
        UUID user_id FK
        UUID workspace_id FK
        VARCHAR(50) resource_type
        UUID resource_id
        INET ip_address
        TEXT user_agent
        JSONB metadata
        TIMESTAMPTZ created_at
    }
    "platform.user_preferences" {
        UUID id PK
        UUID user_id FK
        VARCHAR(50) category
        VARCHAR(255) key
        JSONB value
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.deployment_configs" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) name
        VARCHAR(50) platform
        TEXT repository_url
        VARCHAR(255) branch
        VARCHAR(50) deployment_type
        JSONB environment_variables
        TEXT build_command
        TEXT start_command
        INTEGER port
        VARCHAR(255) health_check_path
        BOOLEAN auto_deploy
        BOOLEAN is_active
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.deployments" {
        UUID id PK
        UUID config_id FK
        INTEGER deployment_number
        VARCHAR(50) status
        VARCHAR(255) commit_sha
        TEXT commit_message
        VARCHAR(255) commit_author
        TEXT build_logs
        TEXT deployment_url
        TIMESTAMPTZ started_at
        TIMESTAMPTZ completed_at
        TEXT error_message
        UUID server_id FK
        JSONB deployment_metadata
    }
    "platform.github_installations" {
        UUID id PK
        UUID workspace_id FK
        INTEGER github_installation_id
        VARCHAR(255) github_account_name
        VARCHAR(50) github_account_type
        TEXT access_token_encrypted
        TEXT refresh_token_encrypted
        TIMESTAMPTZ token_expires_at
        VARCHAR(255) webhook_secret
        JSONB repositories
        JSONB permissions
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.deployment_environments" {
        UUID id PK
        UUID config_id FK
        VARCHAR(50) name
        BOOLEAN is_active
        JSONB resource_limits
        JSONB scaling_config
        VARCHAR(255) custom_domain
        UUID ssl_certificate_id
        JSONB environment_variables
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.deployment_webhook_events" {
        UUID id PK
        INTEGER github_installation_id
        VARCHAR(100) event_type
        JSONB event_payload
        BOOLEAN processed
        TIMESTAMPTZ processed_at
        TEXT error_message
        TIMESTAMPTZ created_at
    }
    "platform.marketplace_servers" {
        UUID id PK
        VARCHAR(255) slug
        VARCHAR(255) name
        TEXT description
        VARCHAR(500) short_description
        VARCHAR(50) category
        VARCHAR(255) author
        TEXT github_url
        INTEGER github_stars
        TIMESTAMPTZ github_last_updated
        TEXT logo_url
        TEXT banner_url
        JSONB deployment_template
        JSONB resource_requirements
        VARCHAR(100) pricing_estimate
        TEXT[] tags
        TEXT[] features
        BOOLEAN is_featured
        BOOLEAN is_verified
        BOOLEAN is_official
        INTEGER install_count
        DECIMAL rating_average
        INTEGER rating_count
        TEXT documentation_url
        TEXT support_url
        VARCHAR(50) license
        VARCHAR(20) min_agentflare_version
        TIMESTAMPTZ last_verified_at
        TEXT verification_notes
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.marketplace_deployments" {
        UUID id PK
        UUID marketplace_server_id FK
        UUID deployment_config_id FK
        UUID workspace_id FK
        VARCHAR(50) version
        TIMESTAMPTZ deployed_at
        TIMESTAMPTZ last_updated_at
        BOOLEAN auto_update_enabled
        JSONB custom_config_overrides
    }
    "platform.marketplace_server_versions" {
        UUID id PK
        UUID marketplace_server_id FK
        VARCHAR(50) version
        TEXT release_notes
        JSONB deployment_template
        BOOLEAN is_stable
        BOOLEAN is_deprecated
        VARCHAR(50) minimum_migration_version
        TIMESTAMPTZ created_at
    }
    "platform.marketplace_reviews" {
        UUID id PK
        UUID marketplace_server_id FK
        UUID user_id FK
        UUID workspace_id FK
        INTEGER rating
        VARCHAR(255) title
        TEXT review
        BOOLEAN is_verified_deployment
        INTEGER helpful_count
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.marketplace_categories" {
        UUID id PK
        VARCHAR(50) slug
        VARCHAR(100) name
        TEXT description
        TEXT icon_url
        INTEGER display_order
        BOOLEAN is_active
        INTEGER server_count
        TIMESTAMPTZ created_at
    }
    "platform.marketplace_collections" {
        UUID id PK
        VARCHAR(255) slug
        VARCHAR(255) name
        TEXT description
        TEXT icon_url
        BOOLEAN is_featured
        INTEGER display_order
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.marketplace_collection_servers" {
        UUID collection_id FK
        UUID marketplace_server_id FK
        INTEGER display_order
        TIMESTAMPTZ added_at
    }
    "platform.workspace_integrations" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(50) type
        VARCHAR(255) name
        TEXT description
        JSONB config
        VARCHAR(20) status
        TEXT status_message
        TIMESTAMPTZ last_sync_at
        TIMESTAMPTZ last_error_at
        TEXT last_error_message
        BOOLEAN is_active
        UUID created_by FK
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "platform.integration_activities" {
        UUID id PK
        UUID integration_id FK
        VARCHAR(50) action
        VARCHAR(20) status
        TEXT details
        UUID performed_by FK
        TIMESTAMPTZ created_at
    }
    "platform.api_keys" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) name
        TEXT description
        api_key_type key_type
        VARCHAR(255) key_hash
        VARCHAR(20) key_prefix
        VARCHAR(4) key_suffix
        VARCHAR(255) key_hint
        api_key_scope[] scopes
        INET[] allowed_ips
        TEXT[] allowed_origins
        INTEGER rate_limit_requests
        INTEGER rate_limit_window_seconds
        INTEGER rate_limit_burst
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ last_used_at
        INET last_used_ip
        TEXT last_used_user_agent
        TIMESTAMPTZ revoked_at
        UUID revoked_by FK
        TEXT revoke_reason
        BIGINT usage_count
        BIGINT error_count
        TIMESTAMPTZ last_error_at
        TEXT last_error_message
        JSONB tags
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        UUID created_by FK
    }
    "platform.api_key_usage_logs" {
        UUID id PK
        UUID api_key_id FK
        TIMESTAMPTZ timestamp
        VARCHAR(100) service_name
        VARCHAR(500) endpoint_path
        VARCHAR(10) http_method
        INTEGER status_code
        INTEGER response_time_ms
        TEXT error_message
        INET ip_address
        TEXT user_agent
        UUID request_id
        BIGINT bytes_in
        BIGINT bytes_out
        INTEGER tokens_used
        DECIMAL cost_estimate
        INTEGER rate_limit_remaining
        TIMESTAMPTZ rate_limit_reset_at
    }
    "platform.api_key_rate_limits" {
        UUID api_key_id PK
        DECIMAL tokens_remaining
        TIMESTAMPTZ last_refill_at
        TIMESTAMPTZ current_window_start
        INTEGER current_window_requests
        INTEGER version
        TIMESTAMPTZ updated_at
    }

    "platform.accounts" }o--|| "auth.users" : "user_id"
    "platform.accounts" }o--|| "auth.providers" : "provider_id"
    "platform.workspaces" }o--|| "auth.users" : "owner_id"
    "platform.logs" }o--|| "auth.users" : "user_id"
    "platform.logs" }o--|| "platform.workspaces" : "workspace_id"
    "platform.user_preferences" }o--|| "auth.users" : "user_id"
    "platform.deployment_configs" }o--|| "platform.workspaces" : "workspace_id"
    "platform.deployments" }o--|| "platform.deployment_configs" : "config_id"
    "platform.deployments" }o--|| "tool_server.registry" : "server_id"
    "platform.github_installations" }o--|| "platform.workspaces" : "workspace_id"
    "platform.deployment_environments" }o--|| "platform.deployment_configs" : "config_id"
    "platform.marketplace_deployments" }o--|| "platform.marketplace_servers" : "marketplace_server_id"
    "platform.marketplace_deployments" }o--|| "platform.deployment_configs" : "deployment_config_id"
    "platform.marketplace_deployments" }o--|| "platform.workspaces" : "workspace_id"
    "platform.marketplace_server_versions" }o--|| "platform.marketplace_servers" : "marketplace_server_id"
    "platform.marketplace_reviews" }o--|| "platform.marketplace_servers" : "marketplace_server_id"
    "platform.marketplace_reviews" }o--|| "auth.users" : "user_id"
    "platform.marketplace_reviews" }o--|| "platform.workspaces" : "workspace_id"
    "platform.marketplace_collection_servers" }o--|| "platform.marketplace_collections" : "collection_id"
    "platform.marketplace_collection_servers" }o--|| "platform.marketplace_servers" : "marketplace_server_id"
    "platform.workspace_integrations" }o--|| "platform.workspaces" : "workspace_id"
    "platform.workspace_integrations" }o--|| "auth.users" : "created_by"
    "platform.integration_activities" }o--|| "platform.workspace_integrations" : "integration_id"
    "platform.integration_activities" }o--|| "auth.users" : "performed_by"
    "platform.api_keys" }o--|| "platform.workspaces" : "workspace_id"
    "platform.api_keys" }o--|| "auth.users" : "revoked_by"
    "platform.api_keys" }o--|| "auth.users" : "created_by"
    "platform.api_key_usage_logs" }o--|| "platform.api_keys" : "api_key_id"
    "platform.api_key_rate_limits" }o--|| "platform.api_keys" : "api_key_id"
```</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/db.contract.platform.md has been created.
