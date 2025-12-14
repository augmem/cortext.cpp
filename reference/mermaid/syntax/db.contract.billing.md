# Database Contract - Billing Module

```mermaid
erDiagram
    "billing.subscriptions" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) stripe_subscription_id
        VARCHAR(255) stripe_customer_id
        plan_kind plan_kind
        subscription_status status
        billing_period billing_period
        TIMESTAMPTZ current_period_start
        TIMESTAMPTZ current_period_end
        TIMESTAMPTZ cancel_at
        TIMESTAMPTZ canceled_at
        TIMESTAMPTZ trial_start
        TIMESTAMPTZ trial_end
        VARCHAR(255) default_payment_method_id
        VARCHAR(255) latest_invoice_id
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        UUID user_id FK
    }
    "billing.subscription_usage" {
        UUID id PK
        UUID workspace_id FK
        plan_kind plan_kind
        INTEGER max_tool_calls_monthly
        INTEGER max_replay_days
        INTEGER max_custom_domains
        INTEGER max_team_members
        INTEGER max_tool_servers
        INTEGER max_agents
        JSONB features
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "billing.subscription_usage_summary" {
        UUID id PK
        UUID workspace_id FK
        usage_period period
        TIMESTAMPTZ period_start
        TIMESTAMPTZ period_end
        INTEGER tool_calls_count
        INTEGER unique_sessions
        INTEGER unique_agents
        BIGINT total_input_tokens
        BIGINT total_output_tokens
        NUMERIC estimated_cost_usd
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "billing.invoices" {
        UUID id PK
        UUID workspace_id FK
        UUID subscription_id FK
        VARCHAR(255) stripe_invoice_id
        VARCHAR(255) invoice_number
        VARCHAR(50) status
        INTEGER amount_due
        INTEGER amount_paid
        VARCHAR(3) currency
        TIMESTAMPTZ period_start
        TIMESTAMPTZ period_end
        TIMESTAMPTZ due_date
        TIMESTAMPTZ paid_at
        VARCHAR(255) payment_method_id
        VARCHAR(255) charge_id
        TEXT invoice_pdf
        TEXT hosted_invoice_url
        JSONB line_items
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "billing.payment_methods" {
        UUID id PK
        UUID workspace_id FK
        VARCHAR(255) stripe_payment_method_id
        VARCHAR(50) type
        VARCHAR(50) brand
        VARCHAR(4) last4
        INTEGER exp_month
        INTEGER exp_year
        VARCHAR(255) billing_name
        VARCHAR(255) billing_email
        JSONB billing_address
        BOOLEAN is_default
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "billing.user_subscriptions" {
        UUID id PK
        UUID user_id FK
        VARCHAR(255) stripe_subscription_id
        subscription_tier tier
        subscription_status status
        INTEGER personal_workspace_limit
        INTEGER team_workspace_limit
        TIMESTAMPTZ current_period_start
        TIMESTAMPTZ current_period_end
        TIMESTAMPTZ cancel_at
        TIMESTAMPTZ canceled_at
        TIMESTAMPTZ trial_start
        TIMESTAMPTZ trial_end
        JSONB metadata
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
    "billing.calculation_errors" {
        UUID id PK
        UUID calculation_id
        VARCHAR(100) error_type
        TEXT error_message
        UUID workspace_id
        UUID user_id
        JSONB context
        TIMESTAMPTZ occurred_at
    }
    "billing.aggregation_jobs" {
        UUID id PK
        VARCHAR(255) job_name
        VARCHAR(20) status
        TIMESTAMPTZ started_at
        TIMESTAMPTZ completed_at
        INTEGER items_processed
        TEXT error_message
        TIMESTAMPTZ created_at
    }
    "billing.usage_tracking" {
        UUID id PK
        UUID workspace_id
        UUID user_id
        VARCHAR(100) resource_type
        VARCHAR(255) resource_id
        NUMERIC usage_amount
        VARCHAR(50) unit
        INTEGER token_count
        NUMERIC cost
        TIMESTAMPTZ tracked_at
    }

    "billing.subscriptions" }o--|| "platform.workspaces" : "workspace_id"
    "billing.subscriptions" }o--|| "auth.users" : "user_id"
    "billing.subscription_usage" }o--|| "platform.workspaces" : "workspace_id"
    "billing.subscription_usage_summary" }o--|| "platform.workspaces" : "workspace_id"
    "billing.invoices" }o--|| "platform.workspaces" : "workspace_id"
    "billing.invoices" }o--|| "billing.subscriptions" : "subscription_id"
    "billing.payment_methods" }o--|| "platform.workspaces" : "workspace_id"
    "billing.user_subscriptions" }o--|| "auth.users" : "user_id"
    "billing.calculation_errors" }o--|| "platform.workspaces" : "workspace_id"
    "billing.calculation_errors" }o--|| "auth.users" : "user_id"
    "billing.usage_tracking" }o--|| "platform.workspaces" : "workspace_id"
    "billing.usage_tracking" }o--|| "auth.users" : "user_id"
```</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/db.contract.billing.md has been created.
