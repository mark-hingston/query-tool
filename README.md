# Embeddings Query Server

An MCP (Model Context Protocol) server for querying vector stores and knowledge graphs created by the [lance-embedder](https://github.com/mark-hingston/lance-embedder) project.

## Features

- **Vector Search**: Query LanceDB vector stores with semantic search
- **Graph Search**: Query knowledge graphs built with GraphRAG for relationship-aware retrieval
- **Auto Mode**: Automatically use graph search when available, fallback to vector search
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
- `--dimensions <number>` - Embedding dimensions (default: 1024)

**Query Settings:**
- `--top-k <number>` - Number of results to return (default: 10)

**Graph Settings:**
- `--enable-graph` - Enable graph search mode (default: disabled for fast startup)
- `--graph-threshold <number>` - Similarity threshold for graph queries (default: 0.7)
- `--random-walk-steps <number>` - Steps for graph traversal (default: 100)
- `--restart-prob <number>` - Restart probability for random walk (default: 0.15)
- `--preload-graph` - Start building graph in background at startup (allows immediate queries via vector search)

**Performance Settings:**
- `--verbose` - Enable verbose logging for debugging and performance monitoring

### Examples

**Basic vector search (fast startup):**
```bash
node dist/index.js \
  --index-path ./embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model \
  --top-k 5
```

**With graph search (lazy loading - builds on first query):**
```bash
node dist/index.js \
  --index-path ./embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model \
  --enable-graph \
  --top-k 5
```

**With background graph preloading (best for production):**
```bash
node dist/index.js \
  --index-path ./embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model \
  --enable-graph \
  --preload-graph \
  --verbose
```

**For large indexes (10K+ files) - recommended settings:**
```bash
node dist/index.js \
  --index-path ./embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model \
  --enable-graph \
  --preload-graph \
  --top-k 10 \
  --verbose
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
- `score` - Similarity score (embedding similarity)
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

## Performance Optimizations

The server implements several optimizations for handling large indexes (10K+ files):

### 1. **Lazy Loading (Default)**
- Graph construction is deferred until the first query
- Server starts instantly, even with large indexes
- First query triggers graph building (may take 30s-2min for 10K files)
- Subsequent queries use the cached graph

### 2. **Parallel Batch Loading**
- All batch files are loaded in parallel using `Promise.all()`
- Significantly reduces I/O time for large datasets
- Progress logging shows loading status for large indexes

### 3. **Automatic Fallback**
- If a query arrives while graph is building, it automatically falls back to vector search
- No query failures due to graph build timing
- Warning logged to console for debugging

### 4. **Persistent Graph Cache**
- Graph metadata is cached to disk after first build
- Future optimizations could serialize the full graph structure
- Currently limited by Mastra GraphRAG serialization API

### 5. **Background Preloading (Optional)**
- Use `--preload-graph` to start building graph at startup
- Server remains responsive via vector search during build
- Graph becomes available when ready, no blocking

### 6. **Worker Thread Support**
- Graph building can be offloaded to worker threads
- Keeps main thread responsive for queries
- Currently experimental (see `graph-worker.ts`)

## How It Works

1. **Startup**: 
   - Validates configuration and loads LanceDB index
   - Loads graph metadata (but not the full graph - lazy loading)
   - Optionally starts background graph build if `--preload-graph` is set
   - Server is ready immediately

2. **Query Processing**: When a query is received:
   - Generates query embedding using the specified model
   - Chooses search mode (vector or graph) based on the `mode` parameter
   - For vector search: queries LanceDB directly (always fast)
   - For graph search: 
     - Checks graph build state
     - If building, falls back to vector search with warning
     - If ready, uses graph for relationship-aware search
     - If idle, triggers lazy build then uses graph
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
     --model text-embedding-model \
     --enable-graph
   ```

## Migration Guide: Before vs After Optimizations

### Before (Slow Startup for Large Indexes)
- Server would build entire graph at startup
- For 10K files: could take 2-5 minutes before server is ready
- No queries could be served during this time
- Graph was always loaded even if not used

### After (Fast Startup + Lazy Loading)
- Server starts instantly (< 1 second)
- Graph is built on first query (or in background with `--preload-graph`)
- Vector search is always immediately available
- Graph search becomes available after build completes
- Parallel loading reduces build time by ~50-70%

### Behavioral Changes

**Without `--enable-graph`:**
- No change - vector search only, instant startup

**With `--enable-graph` (default - lazy loading):**
- Server starts instantly
- First graph query triggers build (30s-2min for large indexes)
- Queries during build automatically fall back to vector search
- Subsequent queries use cached graph

**With `--enable-graph --preload-graph` (recommended for production):**
- Server starts instantly
- Graph building starts in background immediately
- All queries use vector search until graph is ready
- No user-facing delays, graph becomes available when ready

## Error Handling

The server validates on startup:
- Index path exists
- LanceDB table exists
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
