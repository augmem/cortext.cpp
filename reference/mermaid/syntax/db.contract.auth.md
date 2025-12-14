# Database Contract - Authentication Module

```mermaid
erDiagram
    "auth.users" {
        UUID id PK
        VARCHAR(255) auth_id
        VARCHAR(255) email
        VARCHAR(255) name
        TEXT avatar_url
        TEXT image
        VARCHAR(255) password_hash
        TIMESTAMPTZ email_verified
        JSONB preferences
        TIMESTAMPTZ deleted_at
        TIMESTAMPTZ last_active_at
        INTEGER failed_login_attempts
        TIMESTAMPTZ last_failed_login
        TIMESTAMPTZ locked_until
        BOOLEAN two_factor_enabled
        TEXT two_factor_secret
        BOOLEAN two_factor_verified
        TEXT[] two_factor_backup_codes
        TIMESTAMPTZ two_factor_enabled_at
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        VARCHAR(50) username
        subscription_tier subscription_tier
        VARCHAR(255) stripe_customer_id
    }
    "auth.sessions" {
        UUID id PK
        VARCHAR(255) session_token
        UUID user_id FK
        TIMESTAMPTZ expires
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.tokens" {
        UUID id PK
        VARCHAR(255) token
        token_kind kind
        UUID user_id FK
        VARCHAR(255) email
        TIMESTAMPTZ expires
        TIMESTAMPTZ used_at
        TIMESTAMPTZ created_at
    }
    "auth.api_keys" {
        UUID id PK
        VARCHAR(255) key_hash
        VARCHAR(255) name
        UUID workspace_id
        UUID user_id FK
        JSONB permissions
        TIMESTAMPTZ last_used_at
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.providers" {
        UUID id PK
        VARCHAR(255) name
        VARCHAR(255) display_name
        provider_kind kind
        VARCHAR(255) client_id
        TEXT client_secret
        TEXT authorization_url
        TEXT token_url
        TEXT userinfo_url
        TEXT[] scopes
        JSONB configuration
        BOOLEAN is_active
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.webauthn_credentials" {
        UUID id PK
        UUID user_id FK
        BYTEA credential_id
        BYTEA public_key
        VARCHAR(255) name
        BYTEA aaguid
        INTEGER sign_count
        TEXT[] transports
        TIMESTAMPTZ created_at
        TIMESTAMPTZ last_used_at
    }
    "auth.refresh_tokens" {
        UUID id PK
        UUID user_id FK
        VARCHAR(255) token_hash
        UUID family_id
        UUID device_id
        VARCHAR(255) device_name
        INET ip_address
        TEXT user_agent
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ revoked_at
        TIMESTAMPTZ last_used_at
        TIMESTAMPTZ created_at
    }
    "auth.login_attempts" {
        UUID id PK
        UUID user_id FK
        VARCHAR(255) email
        INET ip_address
        TEXT user_agent
        BOOLEAN success
        VARCHAR(255) failure_reason
        TIMESTAMPTZ attempted_at
    }
    "auth.account_lockouts" {
        UUID id PK
        UUID user_id FK
        VARCHAR(255) email
        TIMESTAMPTZ locked_at
        TIMESTAMPTZ locked_until
        VARCHAR(255) unlock_token
        TIMESTAMPTZ unlock_token_expires
        VARCHAR(255) reason
        TIMESTAMPTZ unlocked_at
        VARCHAR(255) unlocked_by
    }
    "auth.audit_log" {
        UUID id PK
        audit_event_kind event_kind
        UUID user_id FK
        VARCHAR(255) email
        INET ip_address
        TEXT user_agent
        VARCHAR(255) device_id
        VARCHAR(255) session_id
        JSONB metadata
        VARCHAR(255) error_code
        TEXT error_message
        TIMESTAMPTZ created_at
    }
    "auth.oauth_states" {
        UUID id PK
        VARCHAR(255) state
        UUID provider_id FK
        TEXT redirect_uri
        VARCHAR(255) code_verifier
        VARCHAR(255) nonce
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ created_at
    }
    "auth.rate_limits" {
        UUID id PK
        VARCHAR(255) identifier
        VARCHAR(50) identifier_type
        VARCHAR(100) action
        INTEGER count
        TIMESTAMPTZ window_start
        INTERVAL window_duration
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.security_incidents" {
        UUID id PK
        VARCHAR(100) incident_type
        VARCHAR(20) severity
        TEXT description
        UUID affected_user_id FK
        INET source_ip
        JSONB metadata
        VARCHAR(50) status
        JSONB auto_actions
        TIMESTAMPTZ resolved_at
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.ip_reputation" {
        UUID id PK
        INET ip_address
        INTEGER reputation_score
        CHAR(2) country_code
        VARCHAR(255) city
        BOOLEAN is_vpn
        BOOLEAN is_tor
        BOOLEAN is_datacenter
        TEXT[] threat_types
        TIMESTAMPTZ last_threat_detected
        TEXT whitelist_reason
        TEXT blacklist_reason
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "auth.blacklisted_tokens" {
        TEXT token_id PK
        TIMESTAMPTZ expires_at
        TIMESTAMPTZ created_at
        TEXT reason
    }

    "auth.sessions" }o--|| "auth.users" : "user_id"
    "auth.tokens" }o--|| "auth.users" : "user_id"
    "auth.api_keys" }o--|| "auth.users" : "user_id"
    "auth.webauthn_credentials" }o--|| "auth.users" : "user_id"
    "auth.refresh_tokens" }o--|| "auth.users" : "user_id"
    "auth.login_attempts" }o--|| "auth.users" : "user_id"
    "auth.account_lockouts" }o--|| "auth.users" : "user_id"
    "auth.audit_log" }o--|| "auth.users" : "user_id"
    "auth.security_incidents" }o--|| "auth.users" : "affected_user_id"
    "auth.oauth_states" }o--|| "auth.providers" : "provider_id"
```</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/db.contract.auth.md has been created.
