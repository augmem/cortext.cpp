# Database Contract Diagrams

This directory contains the AgentFlare database schema documentation organized into manageable modules.

## 📁 File Structure

### Overview Files
- **`db.contract.overview.md`** - High-level system architecture showing module relationships
- **`db.contract.md`** - Complete schema (1797 lines) - use for reference, not daily work

### Core Modules
- **`db.contract.auth.md`** - Authentication, users, sessions, security
- **`db.contract.platform.md`** - Workspaces, deployments, marketplace, integrations

### Feature Modules
- **`db.contract.billing.md`** - Subscriptions, payments, usage tracking
- **`db.contract.agent.md`** - Agent registry, sessions, tool calls
- **`db.contract.proxy.md`** - Route management, API keys, health checks

### Supporting Modules
- **`db.contract.telemetry.md`** - Metrics collection and export
- **`db.contract.queue.md`** - Job processing and scheduling
- **`db.contract.model.md`** - AI model definitions and pricing
- **`db.contract.tool-server.md`** - Tool server registry
- **`db.contract.workspace.md`** - Workspace-specific tables
- **`db.contract.ssl.md`** - SSL certificate management
- **`db.contract.two-factor-auth.md`** - 2FA implementation
- **`db.contract.notifications.md`** - User notifications
- **`db.contract.metrics.md`** - System metrics
- **`db.contract.rule.md`** - Rule engine
- **`db.contract.json.md`** - JSON schema management
- **`db.contract.tokenizer.md`** - Tokenization configs

## 🎯 How to Use

### For New Developers
1. **Start here** → `db.contract.overview.md` - understand the big picture
2. **Pick your module** → Based on your current task (auth, billing, agent, etc.)
3. **Deep dive** → Read the specific module's detailed schema

### For Specific Tasks
- **Adding auth features** → `db.contract.auth.md`
- **Billing integration** → `db.contract.billing.md`
- **Agent development** → `db.contract.agent.md`
- **API proxy work** → `db.contract.proxy.md`

### For Architecture Decisions
- **Cross-module relationships** → `db.contract.overview.md`
- **Complete reference** → `db.contract.md` (large file)

## 🔧 Mermaid ER Diagram Syntax

### Table Definitions
```mermaid
"table.name" {
    UUID id PK
    VARCHAR(255) name
    UUID user_id FK
    TIMESTAMPTZ created_at
}
```

### Relationships
- `}o--||` = Many-to-One (crow's foot to single line)
- `||--o{` = One-to-Many (single line to crow's foot)
- `||--||` = One-to-One
- `}o--o{` = Many-to-Many

### Field Types
- **PK** = Primary Key
- **FK** = Foreign Key
- **UUID** = Universally Unique Identifier
- **VARCHAR(n)** = Variable length string (max n chars)
- **TEXT** = Unlimited text
- **INTEGER** = 32-bit integer
- **BIGINT** = 64-bit integer
- **NUMERIC** = Decimal number
- **BOOLEAN** = True/false
- **JSONB** = JSON data (PostgreSQL)
- **TIMESTAMPTZ** = Timestamp with timezone
- **INET** = IP address
- **BYTEA** = Binary data

## 📝 Notes

- **Simplified Types**: Complex SQL types like `DECIMAL(10,2)` are simplified to `DECIMAL` for Mermaid compatibility
- **Cross-Module Links**: Each module shows internal relationships and references external tables
- **Search-Friendly**: Use IDE search (Cmd+F) to find specific tables or fields across all files
- **Version Control**: These diagrams represent the current schema state

## 🚀 Contributing

When updating the database schema:
1. Update the relevant module file(s)
2. Update the overview if adding new modules or major relationships
3. Consider updating the main `db.contract.md` file if you have the tooling for it
4. Test that all Mermaid diagrams render correctly

## 📊 Statistics

- **Total Tables**: 70+ across all modules
- **Total Relationships**: 120+ foreign key relationships
- **Modules**: 16 focused modules + 1 overview
- **File Size Reduction**: From 1 large file to 16+ manageable modules</contents>
</xai:function_call">The file /Users/gabrielwillen/VSCode/agentflare/.claude/docs/mermaid/syntax/README.md has been created.
