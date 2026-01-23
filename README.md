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
| `memory_set` | Store or update a memory |
| `memory_get` | Retrieve by exact key |
| `memory_delete` | Remove a memory |
| `memory_list` | Browse with pattern/tag filtering |
| `memory_search` | Semantic similarity search |
| `memory_fulltext` | Substring/keyword search |
| `memory_tags` | List all tags with counts |
| `memory_stats` | Storage statistics |
| `memory_embed` | Force embedding regeneration |
| `memory_history` | Version history for a key |

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

## Versioning Notes

- **NumPy**: Pinned to `<2.0.0` due to PyTorch compatibility. PyTorch wheels compiled against NumPy 1.x don't work with NumPy 2.x.
- **sentence-transformers**: Requires `torch` which has the NumPy constraint.

If you see errors like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x", ensure numpy is `<2.0.0`.

## License

MIT
