# Example MCP Client Configuration

This file shows how to configure various MCP clients to use the embeddings query server.

## Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "my-codebase": {
      "command": "node",
      "args": [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/your/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--index-name", "my-codebase",
        "--top-k", "5"
      ]
    }
  }
}
```

**Note:** The `--index-name` parameter helps LLMs understand what index they're querying. It's displayed in the server name and description.

## Cursor / Windsurf

Add to your MCP settings:

```json
{
  "my-codebase": {
    "command": "node",
    "args": [
      "/path/to/query-tool-server/dist/index.js",
      "--index-path", "/path/to/your/embeddings",
      "--base-url", "http://localhost:1234/v1",
      "--model", "text-embedding-model",
      "--index-name", "my-codebase"
    ]
  }
}
```

## With Graph Search Enabled

If you have GraphRAG data (created with `embedder --enable-graph`):

```json
{
  "my-docs": {
    "command": "node",
    "args": [
      "/path/to/query-tool-server/dist/index.js",
      "--index-path", "/path/to/your/embeddings",
      "--base-url", "http://localhost:1234/v1",
      "--model", "text-embedding-model",
      "--index-name", "my-docs",
      "--enable-graph",
      "--graph-threshold", "0.75",
      "--random-walk-steps", "150"
    ]
  }
}
```

## Multiple Indexes

You can run multiple instances of the query server, each pointing to different indexes:

```json
{
  "mcpServers": {
    "frontend-codebase": {
      "command": "node",
      "args": [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/frontend/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--index-name", "frontend-codebase"
      ]
    },
    "backend-codebase": {
      "command": "node",
      "args": [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/backend/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--index-name", "backend-codebase"
      ]
    },
    "documentation": {
      "command": "node",
      "args": [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/docs/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--index-name", "documentation"
      ]
    }
  }
}
```

The LLM will see each server with its descriptive name (e.g., "frontend-codebase Query Server") and understand which index to query for different tasks.

## Using with Mastra MCPClient

If you want to use this server programmatically:

```typescript
import { MCPClient } from "@mastra/mcp";
import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";

const mcp = new MCPClient({
  servers: {
    codebase: {
      command: "node",
      args: [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/your/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--index-name", "my-codebase",
      ],
    },
  },
});

const agent = new Agent({
  name: "RAG Agent",
  instructions: "You can search through the indexed codebase to answer questions.",
  model: openai("gpt-4o"),
  tools: await mcp.getTools(),
});

const response = await agent.generate(
  "How does authentication work in this codebase?"
);
console.log(response.text);
```

## Tool Usage

The MCP server exposes a single tool called `query_index` with these parameters:

- `query` (string, required) - Your search query
- `mode` (enum, optional) - Search mode: "auto" (default), "vector", or "graph"

Example queries:
- "Find code related to user authentication"
- "Show me error handling patterns"
- "What database queries are used?"

The tool will automatically:
1. Generate an embedding for your query
2. Search the vector store (or graph if available)
3. Return the most relevant code chunks

Each result includes:
- `text` - The code or text content
- `source` - File path where it was found
- `score` - Relevance score
- `chunkIndex` - Position within the file
