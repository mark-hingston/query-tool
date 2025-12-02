# Example MCP Client Configuration

This file shows how to configure various MCP clients to use the embeddings query server.

## Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "query-tool": {
      "command": "node",
      "args": [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/your/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
        "--top-k", "5"
      ]
    }
  }
}
```

## Cursor / Windsurf

Add to your MCP settings:

```json
{
  "query-tool": {
    "command": "node",
    "args": [
      "/path/to/query-tool-server/dist/index.js",
      "--index-path", "/path/to/your/embeddings",
      "--base-url", "http://localhost:1234/v1",
      "--model", "text-embedding-model"
    ]
  }
}
```

## With Graph Search Enabled

If you have GraphRAG data (created with `embedder --enable-graph`):

```json
{
  "query-tool": {
    "command": "node",
    "args": [
      "/path/to/query-tool-server/dist/index.js",
      "--index-path", "/path/to/your/embeddings",
      "--base-url", "http://localhost:1234/v1",
      "--model", "text-embedding-model",
      "--graph-threshold", "0.75",
      "--random-walk-steps", "150"
    ]
  }
}
```

## Using with Mastra MCPClient

If you want to use this server programmatically:

```typescript
import { MCPClient } from "@mastra/mcp";
import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";

const mcp = new MCPClient({
  servers: {
    embeddings: {
      command: "node",
      args: [
        "/path/to/query-tool-server/dist/index.js",
        "--index-path", "/path/to/your/embeddings",
        "--base-url", "http://localhost:1234/v1",
        "--model", "text-embedding-model",
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
