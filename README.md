# Embeddings Query Server

An MCP (Model Context Protocol) server for querying vector stores and knowledge graphs created by the [embedder](../lance-embedder) project.

## Features

- **Vector Search**: Query LanceDB vector stores with semantic search
- **Graph Search**: Query knowledge graphs built with GraphRAG for relationship-aware retrieval
- **Auto Mode**: Automatically use graph search when available, fallback to vector search
- **Reranking**: Optional reranking of results for improved relevance
- **Stdio Transport**: Compatible with MCP clients like Claude Desktop, Cursor, and Windsurf

## Installation

```bash
npm install
npm run build
```

## Usage

### Required Arguments

- `--index-path <path>` - Path to the embeddings directory created by embedder
- `--base-url <url>` - LM Studio base URL for generating query embeddings
- `--model <name>` - Embedding model name (must match what was used for indexing)

### Optional Arguments

**Index Settings:**
- `--table-name <name>` - LanceDB table name (default: "embeddings")
- `--dimensions <number>` - Embedding dimensions (default: 2560)

**Query Settings:**
- `--top-k <number>` - Number of results to return (default: 10)

**Graph Settings:**
- `--graph-threshold <number>` - Similarity threshold for graph queries (default: 0.7)
- `--random-walk-steps <number>` - Steps for graph traversal (default: 100)
- `--restart-prob <number>` - Restart probability for random walk (default: 0.15)

**Reranking:**
- `--rerank` - Enable reranking of results (off by default)
- `--rerank-model <name>` - Reranker model name (required if --rerank is enabled)

### Example

```bash
node dist/index.js \
  --index-path ./embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model \
  --top-k 5 \
  --rerank \
  --rerank-model gpt-4o-mini
```

## MCP Tool: query_index

The server exposes a single MCP tool called `query_index`.

### Input Parameters

- `query` (string, required) - The search query text
- `mode` (enum: "auto" | "vector" | "graph", default: "auto") - Search mode
  - `auto`: Use graph search if graph data exists, otherwise vector search
  - `vector`: Force vector-only search via LanceDB
  - `graph`: Force graph search (error if no graph data)

### Output

Returns an array of results sorted by score (highest first), each containing:

- `text` - The chunk content
- `source` - File path where the chunk originated
- `score` - Similarity score (reranker score if reranking enabled, otherwise embedding similarity)
- `chunkIndex` - Position of chunk within the source file

## Project Structure

```
query-tool-server/
├── src/
│   ├── index.ts           # Main entry point and MCP server setup
│   ├── lib/
│   │   └── graph-store.ts # GraphRAG data loading
│   └── types/
│       └── index.ts       # TypeScript type definitions
├── dist/                   # Compiled JavaScript output
├── package.json
├── tsconfig.json
└── README.md
```

## How It Works

1. **Startup**: The server validates configuration and loads the LanceDB index and optional graph data
2. **Query Processing**: When a query is received:
   - Generates query embedding using the specified model
   - Chooses search mode (vector or graph) based on the `mode` parameter
   - For vector search: queries LanceDB directly
   - For graph search: loads GraphRAG and performs random walk traversal
   - Optionally fetches 5x results for reranking
   - Sorts results by score and returns top K

## Integration with Embedder

This server is designed to work with indexes created by the [embedder](../lance-embedder) project:

1. Use embedder to create an index:
   ```bash
   embedder -d ./my-repo -o ./embeddings --enable-graph
   ```

2. Start this query server pointing to that index:
   ```bash
   node dist/index.js \
     --index-path ./embeddings \
     --base-url http://localhost:1234/v1 \
     --model text-embedding-model
   ```

## Error Handling

The server validates on startup:
- Index path exists
- LanceDB table exists
- If `--rerank` is provided, `--rerank-model` must also be provided
- If graph search is requested, graph data must be available

All errors are returned as MCP error responses with descriptive messages.

## Development

```bash
# Run in development mode
npm run dev -- --index-path ./embeddings --base-url http://localhost:1234/v1 --model text-embedding-model

# Build for production
npm run build

# Run built version
npm start -- --index-path ./embeddings --base-url http://localhost:1234/v1 --model text-embedding-model
```

## License

ISC
