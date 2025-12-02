#!/usr/bin/env node
import { MCPClient } from "@mastra/mcp";

async function testQueryTool() {
  console.log("Starting MCP test client...");
  
  const client = new MCPClient({
    servers: {
      "query-tool": {
        command: "node",
        args: [
          "dist/index.js",
          "--index-path", process.env.INDEX_PATH || "./test-index",
          "--base-url", process.env.BASE_URL || "http://localhost:1234/v1",
          "--model", process.env.MODEL || "text-embedding-model",
          "--top-k", "5"
        ],
      },
    },
  });

  try {
    console.log("Connecting to server...");
    await client.connect();
    
    console.log("Getting available tools...");
    const tools = await client.getTools();
    console.log("Available tools:", Object.keys(tools));
    
    console.log("\nCalling query_index tool...");
    const result = await client.callTool("query-tool", "query_index", {
      query: "how to get started setup installation configuration project",
      mode: "auto"
    });
    
    console.log("\n=== Result ===");
    console.log(JSON.stringify(result, null, 2));
    
  } catch (error) {
    console.error("\n=== Error ===");
    console.error(error);
    if (error instanceof Error) {
      console.error("Stack:", error.stack);
    }
  } finally {
    await client.close();
  }
}

testQueryTool();
