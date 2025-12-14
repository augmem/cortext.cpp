# Gauge Specification Syntax Guide

## Critical Format Requirements

**File Extension**: Must be `.spec` (case-sensitive - `.Spec` won't work)

**Basic Structure**: Every Gauge specification follows this exact pattern:
1. **One H1 heading** (`#`) for the specification
2. Optional comments and setup
3. **One or more H2 headings** (`##`) for scenarios  
4. **Steps** starting with `* ` (asterisk + space)

## Complete Example Format

Here's the exact format adapted for JSON-based parameters:

```gauge
# Specification Heading

This is an executable specification file. This file follows markdown syntax.
Every heading in this file denotes a scenario. Every bulleted point denotes a step.

To execute this specification, run
	gauge run specs

* Vowels in English language are "aeiou"

## Vowel counts in single word

tags: single word

* The word "gauge" has "3" vowels

## Vowel counts from test data

tags: data-driven

* Load word data from <file:word-data.json>
* Count vowels for each word in data
* Verify vowel counts match expected results
```

**Corresponding JSON data file (word-data.json):**
```json
{
  "test_words": [
    {"word": "Gauge", "expected_vowels": 3},
    {"word": "Mingle", "expected_vowels": 2}, 
    {"word": "Snap", "expected_vowels": 1},
    {"word": "GoCD", "expected_vowels": 1},
    {"word": "Rhythm", "expected_vowels": 0}
  ]
}
```

## Specification Components

### 1. Specification Heading (REQUIRED)
- Must be H1 format: `# Specification Name`
- Only ONE per file
- Must be the first heading

```gauge
# Search Products
```

### 2. Context Steps (OPTIONAL)
- Steps that run before EVERY scenario in the spec
- Placed after spec heading, before first scenario
- Use `* ` format

```gauge
# User Management

* User is logged in as "admin"
* Navigate to users page

## Create new user
* Click "Add User" button
* Fill user details
```

### 3. Scenarios (REQUIRED - at least one)
- Must be H2 format: `## Scenario Name`
- Each represents one test workflow

```gauge
## Successful login
* Enter username "john"
* Enter password "secret"
* Click login button
* User should be on dashboard
```

### 4. Steps
- Must start with `* ` (asterisk followed by space)
- Parameters in double quotes: `"parameter"`
- Dynamic parameters: `<parameter_name>`

```gauge
* Search for product "iPhone"
* Select user with id <user_id>
* Verify "Success" message is displayed
```

### 5. Tags
Two formats supported:

**Format 1: With Tags: prefix**
```gauge
# Search Products
Tags: search, products, admin

## Successful search  
Tags: happy-path, smoke
```

**Format 2: Direct (as seen in example.spec)**
```gauge
## Vowel counts in single word

tags: single word
```

### 6. JSON Parameters
- Use JSON files for complex data structures
- Reference with `<file:filename.json>` syntax

**JSON data file (users.json):**
```json
{
  "admin_user": {
    "name": "John",
    "email": "john@email.com",
    "role": "admin"
  },
  "regular_user": {
    "name": "Jane", 
    "email": "jane@email.com",
    "role": "user"
  }
}
```

**Using in specification:**
```gauge
# Create Users

## Create user scenario
* Load user data from <file:users.json>
* Create user with profile "admin_user"
* Verify user creation successful
```

### 7. Comments
- Any line that doesn't follow specific syntax
- Used for documentation/readability

```gauge
# User Registration

This specification covers the user registration workflow.
We test both successful and failed registration attempts.

## Successful registration
* Navigate to registration page
* Fill valid user details
```

### 8. Tear Down Steps
- Steps that run after EVERY scenario
- Marked by three or more underscores `___`

```gauge
## Test user creation
* Create user "testuser"
* Verify user exists

___
* Delete user "testuser"
* Clear test data
```

## Parameter Types

### Simple Parameters
```gauge
* Search for "iPhone 13"
* Set quantity to "5"
* Verify price is "$999"
```

### Dynamic Parameters (from JSON data)
```gauge
* Login as <username> with password <password>
* Create user with profile <user_type>
* Navigate to <section> page
```

**Used with JSON configuration:**
```json
{
  "username": "testuser",
  "password": "secret123", 
  "user_type": "admin",
  "section": "dashboard"
}
```

### Special Parameters

**File parameter:**
```gauge
* Upload document <file:test-document.pdf>
* Validate content matches <file:/data/expected.txt>
```

**JSON parameter:**
```gauge
* Import users from <file:users.json>
* Validate configuration <file:/config/app-settings.json>
```

## Complete Working Example

```gauge
# E-commerce Product Search

This specification tests the product search functionality.
All tests require a logged-in user.

* User is logged in as "shopper"
* Navigate to products page

## Search for existing product

tags: smoke, search

* Search for product "iPhone"
* Results should contain "iPhone 13"
* Results should contain "iPhone 14"

## Search with filters

tags: advanced-search

* Load search filters from <file:search-filters.json>
* Search for product "phone" 
* Apply filters from configuration
* Results should match expected criteria

## Search with test data

tags: data-driven

* Load test data from <file:search-test-data.json>
* Execute search with test parameters
* Verify results match expected outcomes

## Search with no results

tags: edge-case

* Search for product "nonexistent-product-xyz"
* Verify "No results found" message is displayed
* Verify search suggestions are shown

___
* Clear all filters
* Logout user
```

## Common Mistakes to Avoid

1. **Wrong file extension**: Use `.spec` not `.Spec` or `.md`
2. **Missing space after asterisk**: Use `* Step` not `*Step`
3. **Multiple spec headings**: Only one `#` heading per file
4. **Wrong heading levels**: Scenarios must use `##` not `#` or `###`
5. **Incorrect parameter syntax**: Use `"quotes"` for strings, `<brackets>` for dynamic
6. **Invalid JSON files**: Ensure JSON files are valid and properly formatted
7. **Missing empty lines**: Structure requires proper spacing between sections

## Key Syntax Rules

- **File must end with `.spec`**
- **One H1 heading** (`#`) for specification
- **H2 headings** (`##`) for scenarios  
- **Steps start with `* `** (asterisk + space)
- **String parameters** in `"double quotes"`
- **Dynamic parameters** in `<angle_brackets>`
- **JSON data** using `<file:filename.json>` format
- **Tags** can use `Tags: tag1, tag2` or just `tags: tag1, tag2`
- **Comments** are any non-syntax lines
- **Context steps** go after spec heading, before scenarios
- **Tear down** marked with `___` (three+ underscores)

This syntax must be followed exactly for Gauge to parse and execute the specifications correctly.