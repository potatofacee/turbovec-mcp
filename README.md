# turbovec-mcp

Local semantic code search over MCP, backed by
[turbovec](https://github.com/RyanCodrai/turbovec) (Google's TurboQuant: ~16x
vector compression, fast on Apple Silicon). Bring your own OpenAI-compatible
embeddings endpoint - nothing leaves your machine.

```
code -> chunker -> [your embeddings endpoint] -> turbovec index -> search
```

- **embedder**: any OpenAI-compatible `/v1/embeddings` (a llama.cpp server, etc).
  Dimension is auto-detected.
- **store**: turbovec `IdMapIndex`, persisted per-repo under `.turbovec/`.
- **orchestrator**: this MCP server (chunk, embed, index, search).

## Install / run

```sh
uvx turbovec-mcp        # run directly, no install (recommended)
# or
pipx install turbovec-mcp
```

Requires a running embeddings endpoint. Example with llama.cpp:

```sh
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --pooling mean --port 8081
```

## Configure (environment)

| var | default | meaning |
|-----|---------|---------|
| `TURBOVEC_EMBED_ENDPOINT` | `http://127.0.0.1:8081/v1` | OpenAI-compatible base URL |
| `TURBOVEC_EMBED_MODEL` | `nomic-embed-text-v1.5.Q8_0.gguf` | model name sent in the request |
| `TURBOVEC_EMBED_API_KEY` | `sk-local` | bearer token (unused locally, must be non-empty) |
| `TURBOVEC_DOC_PREFIX` | "" | prefix for documents (nomic: `search_document: `) |
| `TURBOVEC_QUERY_PREFIX` | "" | prefix for queries (nomic: `search_query: `) |
| `TURBOVEC_BATCH_SIZE` | `64` | embedding inputs per request |
| `TURBOVEC_TIMEOUT` | `120` | embedding request timeout (seconds) |
| `TURBOVEC_MAX_EMBED_CHARS` | `1800` | hard char cap per embedded input |
| `TURBOVEC_BIT_WIDTH` | `4` | turbovec quantization bits (2 or 4) |
| `TURBOVEC_CHUNK_LINES` | `60` | lines per chunk |
| `TURBOVEC_CHUNK_OVERLAP` | `12` | overlap between chunks |
| `TURBOVEC_MAX_FILE_MB` | `2` | skip files larger than this |
| `TURBOVEC_EXTRA_EXTENSIONS` | "" | comma list of extra extensions to index |
| `TURBOVEC_EXTRA_SKIP_DIRS` | "" | comma list of extra dirs to skip |

## Register with an MCP client

opencode (`opencode.jsonc`):

```jsonc
"mcp": {
  "turbovec": {
    "type": "local",
    "command": ["uvx", "turbovec-mcp"],
    "environment": {
      "TURBOVEC_EMBED_ENDPOINT": "http://127.0.0.1:8081/v1",
      "TURBOVEC_EMBED_MODEL": "nomic-embed-text-v1.5.Q8_0.gguf",
      "TURBOVEC_QUERY_PREFIX": "search_query: ",
      "TURBOVEC_DOC_PREFIX": "search_document: "
    },
    "enabled": true
  }
}
```

Claude Code:

```sh
claude mcp add turbovec -- uvx turbovec-mcp
```

## Chunking

AST-aware via [tree-sitter](https://tree-sitter.github.io/). Each function /
class / method that fits the embedder becomes one chunk on clean boundaries; a
god-class too large to embed whole is split into its methods; any lines not
covered by a definition (imports, top-level code, a giant leaf function) are
line-windowed, so coverage is total. Unsupported languages / parse failures fall
back to plain line-window chunking - nothing is skipped.

Supported out of the box: Python, JS/TS/TSX, Go, Rust, Java, Kotlin, Scala,
Swift, C/C++, C#, Ruby, PHP, Lua, Perl, R, Julia, Bash, Fortran (f90/f95).
Add more file extensions with `TURBOVEC_EXTRA_EXTENSIONS`.

## Tools

Workflow is two steps: `tv_search` for terse triage, then `tv_fetch` to read
the locations you picked.

- `tv_search(query, k=10, path=".")` - semantic search. Terse: each hit is
  `score` + location (`path:start-end`) + the chunk's signature line. No source
  bodies.
- `tv_fetch(locations, path=".")` - full source for a list of `"path:start-end"`
  locations (the ones `tv_search` returned).
- `tv_index(path=".")` - (re)build the index for a repo. Run once before
  searching, and after large changes.
- `tv_status(path=".")` - whether a repo is indexed + basic stats.

The index lives in `<repo>/.turbovec/` - add it to `.gitignore`.

## File selection

`<repo>/.turbovec/config.json` is the source of truth for which files get
indexed: its `include` / `exclude` glob lists decide everything. Seed it with:

```sh
turbovec-mcp init [path]     # writes .turbovec/config.json (-f to regenerate)
```

`init` seeds `exclude` from the default skip dirs plus any nested `.gitignore`s
in the tree. Edit the globs, then run `turbovec-mcp index`.

## CLI

For testing from the terminal:

```sh
turbovec-mcp init [path]                  # seed .turbovec/config.json
turbovec-mcp index <path>                 # build/rebuild the index
turbovec-mcp status <path>
turbovec-mcp search <path> <query> [-k N] # terse: score  path:lines  signature
turbovec-mcp fetch <path> <location>...   # full source for path:start-end
```

## License

BSD-3-Clause
