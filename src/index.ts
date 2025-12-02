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
  )
  .option("--rerank", "Enable reranking of results", false)
  .option("--rerank-model <name>", "Reranker model name");

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
  graphThreshold: parseFloat(options.graphThreshold),
  randomWalkSteps: parseInt(options.randomWalkSteps),
  restartProb: parseFloat(options.restartProb),
  rerank: options.rerank,
  rerankModel: options.rerankModel,
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

if (config.rerank && !config.rerankModel) {
  console.error("Error: --rerank-model is required when --rerank is enabled");
  process.exit(1);
}

// Initialize OpenAI client for embeddings
const openai = createOpenAI({
  apiKey: "not-needed",
  baseURL: config.baseUrl,
});

// Initialize LanceDB - will be created async in main function
let lanceDb: LanceVectorStore | null = null;

// Load graph store (if it exists)
const graphStore = new GraphStore(config.indexPath);
const hasGraphData = graphStore.load() && graphStore.hasData();

if (hasGraphData) {
  console.log("Graph data found and loaded");
  const stats = graphStore.getStats();
  if (stats) {
    console.log(`Graph contains ${stats.nodeCount} nodes`);
  }
} else {
  console.log("No graph data found - graph search mode will not be available");
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
      // Determine which search mode to use
      let useGraph = false;
      if (mode === "graph") {
        if (!hasGraphData) {
          throw new Error("Graph search requested but no graph data available");
        }
        useGraph = true;
      } else if (mode === "auto") {
        useGraph = hasGraphData;
      }

      // Generate query embedding
      const { embedding: queryEmbedding } = await embed({
        model: openai.embedding(config.model),
        value: query,
      });

      let results: QueryResult[] = [];

      if (useGraph) {
        // Graph-based search
        console.log("Using graph search mode");
        const graphRag = graphStore.buildGraphRAG(
          config.dimensions,
          config.graphThreshold
        );

        if (!graphRag) {
          throw new Error("Failed to build GraphRAG instance");
        }

        const graphResults = await graphRag.query({
          query: queryEmbedding,
          topK: config.rerank ? config.topK * 5 : config.topK,
          randomWalkSteps: config.randomWalkSteps,
          restartProb: config.restartProb,
        });

        results = graphResults.map((node: any) => ({
          text: node.content,
          source: node.metadata.source,
          score: node.score,
          chunkIndex: node.metadata.chunkIndex,
        }));
      } else {
        // Vector-based search
        console.log("Using vector search mode");
        if (!lanceDb) {
          throw new Error("LanceDB not initialized");
        }

        const vectorResults = await lanceDb.query({
          indexName: config.tableName,
          tableName: config.tableName,
          queryVector: queryEmbedding,
          topK: config.rerank ? config.topK * 5 : config.topK,
          includeVector: false,
        });

        results = vectorResults.map((result: any) => ({
          text: result.document || result.text || "",
          source: result.metadata?.source || result.source || "",
          score: result.score || 0,
          chunkIndex: result.metadata?.chunkIndex || result.chunkIndex || 0,
        }));
      }

      // Apply reranking if enabled
      if (config.rerank && config.rerankModel) {
        console.log("Applying reranking");
        // Simple reranking by sorting - in production you'd use a reranker model
        results.sort((a, b) => b.score - a.score);
        results = results.slice(0, config.topK);
      }

      // Sort by score (highest first)
      results.sort((a, b) => b.score - a.score);

      return {
        results,
      };
    } catch (error) {
      console.error("Query error:", error);
      throw error;
    }
  },
});

// Main async function to initialize and start the server
async function main() {
  // Initialize LanceDB
  lanceDb = await LanceVectorStore.create(lanceDbPath);

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

  console.log("Starting MCP server with stdio transport...");
  console.log(`Index path: ${config.indexPath}`);
  console.log(`Table name: ${config.tableName}`);
  console.log(`Model: ${config.model}`);
  console.log(`Dimensions: ${config.dimensions}`);
  console.log(`Top K: ${config.topK}`);
  console.log(`Graph available: ${hasGraphData}`);
  console.log(`Reranking: ${config.rerank ? "enabled" : "disabled"}`);

  // Start the server
  await server.startStdio();
}

// Start the main function
main().catch((error) => {
  console.error("Failed to start MCP server:", error);
  process.exit(1);
});
