# Quick Start Guide

Get the embeddings query server running in 5 minutes.

## Prerequisites

1. Node.js 18+ installed
2. An existing embeddings index created by the [embedder](../lance-embedder) project
3. LM Studio (or compatible) running with an embedding model

## Step 1: Install Dependencies

```bash
cd query-tool-server
npm install
```

## Step 2: Build the Project

```bash
npm run build
```

## Step 3: Test the Server

Run the server with your embeddings:

```bash
node dist/index.js \
  --index-path /path/to/your/embeddings \
  --base-url http://localhost:1234/v1 \
  --model text-embedding-model
```

You should see output like:

```
Graph data found and loaded
Graph contains 1234 nodes
Starting MCP server with stdio transport...
Index path: /path/to/your/embeddings
Table name: embeddings
Model: text-embedding-model
Dimensions: 1024
Top K: 10
Graph available: true
```

## Step 4: Configure Your MCP Client

### For Claude Desktop

1. Open `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add the server configuration:

```json
{
  "mcpServers": {
    "query-tool": {
      "command": "node",
      "args": [
        "/absolute/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/your/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model"
      ]
    }
  }
}
```

3. Restart Claude Desktop
4. You should see the `query_index` tool available

### For Cursor/Windsurf

Follow similar steps in your editor's MCP configuration.

## Step 5: Try It Out

In Claude Desktop (or your MCP client), try asking:

> "Can you search the codebase for authentication code?"

The AI will use the `query_index` tool to search your embeddings and provide relevant results.

## Next Steps

### Enable Graph Search

If you want relationship-aware search, re-index with graph support:

```bash
# In the embedder project
embedder -d ./my-repo -o ./embeddings --enable-graph
```

Then start the query server normally - it will detect and use the graph data automatically.

### Customize Search Parameters

Adjust these for your use case:

- `--top-k 5` - Return fewer results for faster responses
- `--graph-threshold 0.8` - Higher threshold = stricter graph connections
- `--random-walk-steps 200` - More steps = deeper graph exploration

## Troubleshooting

### "Index path does not exist"

Make sure you've created an embeddings index first:

```bash
cd lance-embedder
embedder -d ./your-repo -o ./embeddings
```

### "LanceDB directory not found"

The embeddings directory should contain a `lance/` subdirectory. Make sure the embedder completed successfully.

### "No graph data found"

This is normal if you didn't use `--enable-graph` when creating the index. The server will work fine with vector search only.

### Server doesn't respond

Make sure:
1. LM Studio is running
2. The embedding model is loaded
3. The `--base-url` matches your LM Studio URL

## Example Queries

Once configured, try these queries with your MCP client:

- "Find all database query code"
- "Show me error handling patterns"
- "What API endpoints are defined?"
- "Search for authentication logic"
- "Find code related to user management"

The tool will return relevant chunks from your indexed codebase!
