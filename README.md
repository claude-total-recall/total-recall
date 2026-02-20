# Total Recall

![Total Recall Logo](images/logo.jpg)

A Python MCP server for contextual memory storage with semantic search.

## Features

- **Simple** - Minimal dependencies, single SQLite file for storage
- **Portable** - No external services required, runs entirely local
- **Semantic** - Find memories by meaning, not just exact keys
- **Flexible** - Store anything from one-liners to full documents

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .
```

## Usage

### As MCP Server

Add to your Claude configuration:

```json
{
  "mcpServers": {
    "total-recall": {
      "command": "/path/to/total_recall/venv/bin/python",
      "args": ["-m", "total_recall.server"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `memory_set` | Store or update a memory (clobber guard enforced) |
| `memory_set_from_file` | Store file contents verbatim (bypasses agent summarization) |
| `memory_get` | Retrieve by exact key |
| `memory_delete` | Remove a memory |
| `memory_list` | Browse with pattern/tag filtering |
| `memory_search` | Semantic similarity search |
| `memory_fulltext` | Substring/keyword search |
| `memory_tags` | List all tags with counts |
| `memory_stats` | Storage statistics |
| `memory_embed` | Force embedding regeneration |
| `memory_history` | Version history for a key |

### Clobber Guard

`memory_set` includes a write guard that prevents AI callers from accidentally overwriting rich content with shorter rewrites.

**How it works:** When updating an existing key, if the caller hasn't read it (via `memory_get`) since its last update AND the new content is smaller, the write is **blocked**. The response includes `blocked: true`, `success: false`, and `previous_value` containing the existing content so the caller can construct a proper merge.

| Scenario | Result |
|----------|--------|
| New key | Allowed |
| Read since last write, any size | Allowed |
| Not read, content growing | Allowed (with warning) |
| Not read, content shrinking | **Blocked** |

This uses the existing `accessed_at` / `updated_at` timestamps — no schema changes required. The existing >50% truncation warning still fires as a soft warning after the guard passes.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TOTAL_RECALL_DB` | `~/.total_recall/memory.db` | Database path |
| `TOTAL_RECALL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOTAL_RECALL_AUTO_EMBED` | `true` | Embed on write |
| `TOTAL_RECALL_HISTORY` | `true` | Track version history |
| `TOTAL_RECALL_MAX_HISTORY` | `50` | Max versions per key |

## Key Naming Convention

Use hierarchical dot notation:

```
user.preferences.code_style
project.myapp.conventions
project.myapp.architecture
reference.api_endpoints
```

## Effective Use with Claude

To get the most out of Total Recall, add instructions to your `CLAUDE.md` file. Here's a recommended starting point:

```markdown
## Total Recall - Persistent Memory

Use the total-recall MCP to persist knowledge across context resets. **Proactively save** things worth remembering.

**When to save:**
- User states a preference or convention
- Architectural decisions are made
- Non-obvious codebase discoveries (gotchas, patterns, entry points)
- Multi-session task state and progress
- Anything the user might ask about later or need to reference

**Key naming convention** (dot notation):
- `user.preferences.*` - coding style, communication, tooling prefs
- `project.<name>.*` - architecture, conventions, gotchas for a specific project
- `task.<name>.*` - status, blockers, next steps for ongoing work
- `reference.*` - API endpoints, file locations, commands

**Tools:**
- `memory_set` - store/update (auto-embeds for semantic search)
- `memory_set_from_file` - store file contents verbatim (MCP reads file directly)
- `memory_get` - retrieve by exact key
- `memory_delete` - remove a memory
- `memory_list` - browse with pattern (`project.myapp.*`)
- `memory_search` - find by meaning ("how does auth work")
- `memory_fulltext` - substring/keyword search (when you know exact terms)
- `memory_tags` - list all tags with counts
- `memory_stats` - storage statistics
- `memory_embed` - force (re)generate embeddings
- `memory_history` - view previous versions of a key

**On session start:** Consider `memory_search` for relevant context if resuming work or if the user's request might relate to stored knowledge.

**When uncertain:** If you don't remember something the user expects you to know, or if they reference past work/decisions, search Total Recall before asking them to repeat themselves.

**Update rule:** When updating a Total Recall entry, always preserve existing detail level. Merge new information into the existing content rather than replacing it with a summary.
```

Adapt the key naming conventions to fit your workflow. The key insight is instructing Claude to **proactively** save and retrieve — without this, the tool sits idle.

## Claude Code Hook Integration

Total Recall includes a hook that automatically injects relevant memories into your prompts before Claude processes them. This provides context without you needing to ask Claude to search.

### How It Works

The `total-recall-hook` command:
1. Reads your prompt from stdin (JSON format from Claude Code)
2. Extracts search terms (ticket IDs like `PROJ-123`, significant words)
3. Searches memories: key patterns → semantic embeddings → fulltext
4. Returns matching memories as `additionalContext` for Claude to see

### Configuration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/total_recall/.venv/bin/total-recall-hook"
      }]
    }]
  }
}
```

Replace `/path/to/total_recall` with your actual installation path.

Alternatively, invoke via Python module:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/total_recall/.venv/bin/python -m total_recall.hook"
      }]
    }]
  }
}
```

### What Gets Injected

The hook searches in priority order:

1. **Key pattern** (`match="key"`) - matches terms against memory keys
2. **Semantic** (`match="semantic"`) - embedding similarity ≥ 0.3
3. **Fulltext** (`match="fulltext"`) - substring match in values

Results are formatted as structured XML:

```xml
<total-recall-context>
<memory key="project.myapp.architecture" match="semantic" score="0.72" truncated="true" full_length="4500">
Memory content (truncated to 512 chars)...
</memory>

<memory key="user.preferences.coding-style" match="key">
Short memories appear in full without truncation.
</memory>
</total-recall-context>
```

**Attributes:**
- `key` - memory key (use with `memory_get` for full content)
- `match` - how it was found: `key`, `semantic`, or `fulltext`
- `score` - similarity score (semantic matches only)
- `truncated` / `full_length` - present if content was cut (max 512 chars)

This context is visible to Claude when processing your prompt, giving it relevant memories automatically.

## Versioning Notes

- **NumPy**: Pinned to `<2.0.0` due to PyTorch compatibility. PyTorch wheels compiled against NumPy 1.x don't work with NumPy 2.x.
- **sentence-transformers**: Requires `torch` which has the NumPy constraint.

If you see errors like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x", ensure numpy is `<2.0.0`.

## License

MIT
