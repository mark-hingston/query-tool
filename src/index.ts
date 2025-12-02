#!/usr/bin/env node
import { Command } from "commander";
import { MCPServer } from "@mastra/mcp";
import { createTool } from "@mastra/core/tools";
import { createOpenAI } from "@ai-sdk/openai";
import { embed } from "ai";
import { z } from "zod";
import { LanceVectorStore } from "@mastra/lance";
import * as fs from "fs";
import * as path from "path";
import type { ServerConfig, QueryInput, QueryResult } from "./types/index.js";
import { GraphStore } from "./lib/graph-store.js";

// Parse command line arguments
const program = new Command();

program
  .name("query-tool-server")
  .description("MCP server for querying vector stores and knowledge graphs")
  .version("1.0.0")
  .requiredOption("--index-path <path>", "Path to the embeddings directory")
  .requiredOption("--base-url <url>", "LM Studio base URL for embeddings")
  .requiredOption("--model <name>", "Embedding model name")
  .option("--table-name <name>", "LanceDB table name", "embeddings")
  .option("--dimensions <number>", "Embedding dimensions", "2560")
  .option("--top-k <number>", "Number of results to return", "10")
  .option(
    "--enable-graph",
    "Enable graph search (WARNING: slow startup with large indexes)",
    false
  )
  .option(
    "--graph-threshold <number>",
    "Similarity threshold for graph queries",
    "0.7"
  )
  .option(
    "--random-walk-steps <number>",
    "Steps for graph traversal",
    "100"
  )
  .option(
    "--restart-prob <number>",
    "Restart probability for graph random walk",
    "0.15"
  );

program.parse();

const options = program.opts();

// Build config
const config: ServerConfig = {
  indexPath: options.indexPath,
  baseUrl: options.baseUrl,
  model: options.model,
  tableName: options.tableName,
  dimensions: parseInt(options.dimensions),
  topK: parseInt(options.topK),
  enableGraph: options.enableGraph || false,
  graphThreshold: parseFloat(options.graphThreshold),
  randomWalkSteps: parseInt(options.randomWalkSteps),
  restartProb: parseFloat(options.restartProb),
};

// Validation
if (!fs.existsSync(config.indexPath)) {
  console.error(`Error: Index path does not exist: ${config.indexPath}`);
  process.exit(1);
}

// Check for LanceDB directory - try both possible names
let lanceDbPath = path.join(config.indexPath, "embeddings.lance");
if (!fs.existsSync(lanceDbPath)) {
  // Try legacy name
  lanceDbPath = path.join(config.indexPath, "lance");
  if (!fs.existsSync(lanceDbPath)) {
    console.error(`Error: LanceDB directory not found at:`);
    console.error(`  - ${path.join(config.indexPath, "embeddings.lance")}`);
    console.error(`  - ${path.join(config.indexPath, "lance")}`);
    process.exit(1);
  }
}



// Initialize OpenAI client for embeddings
const openai = createOpenAI({
  apiKey: "not-needed",
  baseURL: config.baseUrl,
});

// Initialize LanceDB and GraphRAG - will be created async in main function
let lanceDb: LanceVectorStore | null = null;
let graphRag: any | null = null;

// Load graph store (if it exists)
const graphStore = new GraphStore(config.indexPath);
const hasGraphData = graphStore.load() && graphStore.hasData();

if (hasGraphData) {
  console.error("Graph data found and loaded");
  const stats = graphStore.getStats();
  if (stats) {
    console.error(`Graph contains ${stats.nodeCount} nodes`);
  }
} else {
  console.error("No graph data found - graph search mode will not be available");
}

// Create the query_index tool
const queryIndexTool = createTool({
  id: "query_index",
  description: "Search the indexed embeddings using vector or graph-based search",
  inputSchema: z.object({
    query: z.string().describe("The search query text"),
    mode: z
      .enum(["auto", "vector", "graph"])
      .default("auto")
      .describe(
        "Search mode: auto (use graph if available), vector (force vector search), or graph (force graph search)"
      ),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        text: z.string(),
        source: z.string(),
        score: z.number(),
        chunkIndex: z.number(),
      })
    ),
  }),
  execute: async ({ context }: { context: QueryInput }) => {
    const { query, mode } = context;

    try {
      console.error(`[Query] Received query: "${query.substring(0, 100)}${query.length > 100 ? '...' : ''}"`);
      console.error(`[Query] Mode: ${mode}`);
      
      // Determine which search mode to use
      let useGraph = false;
      if (mode === "graph") {
        if (!graphRag) {
          throw new Error("Graph search requested but graph is not initialized. Use --enable-graph to enable graph search.");
        }
        useGraph = true;
      } else if (mode === "auto") {
        // Only use graph if it's actually initialized
        useGraph = graphRag !== null;
      }

      console.error(`[Query] Search mode: ${useGraph ? 'graph' : 'vector'}`);

      // Generate query embedding
      console.error(`[Query] Generating embedding for query...`);
      const startEmbed = Date.now();
      const { embedding: queryEmbedding } = await embed({
        model: openai.embedding(config.model),
        value: query,
      });
      console.error(`[Query] Embedding generated in ${Date.now() - startEmbed}ms`);

      let results: QueryResult[] = [];

      if (useGraph) {
        // Graph-based search
        console.error("[Query] Using graph search mode");
        if (!graphRag) {
          throw new Error("GraphRAG not initialized");
        }

        console.error(`[Query] Querying graph with topK=${config.topK}, randomWalkSteps=${config.randomWalkSteps}...`);
        const startGraph = Date.now();
        const graphResults = await graphRag.query({
          query: queryEmbedding,
          topK: config.topK,
          randomWalkSteps: config.randomWalkSteps,
          restartProb: config.restartProb,
        });
        console.error(`[Query] Graph search completed in ${Date.now() - startGraph}ms, found ${graphResults.length} results`);

        results = graphResults.map((node: any) => ({
          text: node.content,
          source: node.metadata.source,
          score: node.score,
          chunkIndex: node.metadata.chunkIndex,
        }));
      } else {
        // Vector-based search
        console.error("[Query] Using vector search mode");
        if (!lanceDb) {
          throw new Error("LanceDB not initialized");
        }

        console.error(`[Query] Querying vector store with topK=${config.topK}...`);
        const startVector = Date.now();
        const vectorResults = await lanceDb.query({
          indexName: config.tableName,
          tableName: config.tableName,
          queryVector: queryEmbedding,
          topK: config.topK,
          includeVector: false,
        });
        console.error(`[Query] Vector search completed in ${Date.now() - startVector}ms, found ${vectorResults.length} results`);

        results = vectorResults.map((result: any) => ({
          text: result.document || result.text || "",
          source: result.metadata?.source || result.source || "",
          score: result.score || 0,
          chunkIndex: result.metadata?.chunkIndex || result.chunkIndex || 0,
        }));
      }

      // Sort by score (highest first)
      results.sort((a, b) => b.score - a.score);

      console.error(`[Query] Returning ${results.length} results`);
      
      return {
        results,
      };
    } catch (error) {
      console.error("[Query] Error:", error);
      throw error;
    }
  },
});

// Main async function to initialize and start the server
async function main() {
  console.error("[Init] Initializing MCP server...");
  
  // Initialize LanceDB
  console.error("[Init] Connecting to LanceDB...");
  const startLance = Date.now();
  lanceDb = await LanceVectorStore.create(lanceDbPath);
  console.error(`[Init] LanceDB connected in ${Date.now() - startLance}ms`);

  // Build GraphRAG once at startup if enabled and graph data exists
  if (config.enableGraph && hasGraphData) {
    console.error("[Init] Building GraphRAG instance (this may take several minutes for large indexes)...");
    const startGraph = Date.now();
    graphRag = graphStore.buildGraphRAG(
      config.dimensions,
      config.graphThreshold
    );
    if (graphRag) {
      console.error(`[Init] GraphRAG built in ${Date.now() - startGraph}ms`);
    } else {
      console.warn("[Init] Failed to build GraphRAG, graph search will be disabled");
    }
  } else if (!config.enableGraph && hasGraphData) {
    console.error("[Init] Graph data available but --enable-graph not specified, using vector search only");
  }

  // Create MCP Server
  const server = new MCPServer({
    name: "Embeddings Query Server",
    version: "1.0.0",
    description:
      "Query vector stores and knowledge graphs created by the embedder",
    tools: {
      query_index: queryIndexTool,
    },
  });

  console.error("\n=== MCP Server Ready ===");
  console.error(`Index path: ${config.indexPath}`);
  console.error(`Table name: ${config.tableName}`);
  console.error(`Model: ${config.model}`);
  console.error(`Dimensions: ${config.dimensions}`);
  console.error(`Top K: ${config.topK}`);
  console.error(`Graph available: ${hasGraphData && graphRag !== null}`);
  console.error("========================\n");

  // Start the server
  await server.startStdio();
}

// Start the main function
main().catch((error) => {
  console.error("Failed to start MCP server:", error);
  process.exit(1);
});
