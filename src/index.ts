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
  .option("--index-name <name>", "Descriptive name for this index (e.g., 'my-codebase', 'documentation')")
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
  )
  .option(
    "--verbose",
    "Enable verbose logging output",
    false
  );

program.parse();

const options = program.opts();

// Build config
const config: ServerConfig = {
  indexPath: options.indexPath,
  baseUrl: options.baseUrl,
  model: options.model,
  indexName: options.indexName,
  tableName: options.tableName,
  dimensions: parseInt(options.dimensions),
  topK: parseInt(options.topK),
  enableGraph: options.enableGraph || false,
  graphThreshold: parseFloat(options.graphThreshold),
  randomWalkSteps: parseInt(options.randomWalkSteps),
  restartProb: parseFloat(options.restartProb),
  verbose: options.verbose || false,
};

// Logging helper - only logs if verbose mode is enabled
const log = (...args: any[]) => {
  if (config.verbose) {
    console.error(...args);
  }
};

// Validation
if (!fs.existsSync(config.indexPath)) {
  console.error(`Error: Index path does not exist: ${config.indexPath}`);
  process.exit(1);
}

// Check for LanceDB directory - determine the correct table name
// Note: Mastra's LanceDB automatically appends .lance to table names
let lanceDbPath = config.indexPath;
let actualTableName = config.tableName;

// Check if embeddings.lance exists (standard name)
const embeddingsLancePath = path.join(config.indexPath, "embeddings.lance");
if (fs.existsSync(embeddingsLancePath)) {
  // Use just "embeddings" - Mastra will append .lance automatically
  actualTableName = "embeddings";
} else {
  // Try legacy name without .lance extension
  const legacyPath = path.join(config.indexPath, "lance");
  if (fs.existsSync(legacyPath)) {
    // Remove .lance extension if present since Mastra adds it
    actualTableName = "lance".replace(/\.lance$/, "");
  } else {
    console.error(`Error: LanceDB directory not found at:`);
    console.error(`  - ${embeddingsLancePath}`);
    console.error(`  - ${legacyPath}`);
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
  log("Graph data found and loaded");
  const stats = graphStore.getStats();
  if (stats) {
    log(`Graph contains ${stats.nodeCount} nodes`);
  }
} else {
  log("No graph data found - graph search mode will not be available");
}

// Create the query_index tool - note: description will be static, but server description is dynamic
const queryIndexTool = createTool({
  id: "query_index",
  description: "Search indexed embeddings using vector or graph-based search. Check the server description to see which index this queries.",
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
  execute: async (executionContext) => {
    log("[Query] Execute function called");
    log("[Query] executionContext keys:", Object.keys(executionContext));
    log("[Query] executionContext.context:", JSON.stringify(executionContext.context));

    const { query, mode } = executionContext.context as QueryInput;

    try {
      log(`[Query] Received query: "${query.substring(0, 100)}${query.length > 100 ? '...' : ''}"`);
      log(`[Query] Mode: ${mode}`);
      
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

      log(`[Query] Search mode: ${useGraph ? 'graph' : 'vector'}`);

      // Generate query embedding
      log(`[Query] Generating embedding for query...`);
      const startEmbed = Date.now();
      const { embedding: queryEmbedding } = await embed({
        model: openai.embedding(config.model),
        value: query,
      });
      log(`[Query] Embedding generated in ${Date.now() - startEmbed}ms`);

      let results: QueryResult[] = [];

      if (useGraph) {
        // Graph-based search
        log("[Query] Using graph search mode");
        if (!graphRag) {
          throw new Error("GraphRAG not initialized");
        }

        log(`[Query] Querying graph with topK=${config.topK}, randomWalkSteps=${config.randomWalkSteps}...`);
        const startGraph = Date.now();
        const graphResults = await graphRag.query({
          query: queryEmbedding,
          topK: config.topK,
          randomWalkSteps: config.randomWalkSteps,
          restartProb: config.restartProb,
        });
        log(`[Query] Graph search completed in ${Date.now() - startGraph}ms, found ${graphResults.length} results`);

        results = graphResults.map((node: any) => ({
          text: node.content,
          source: node.metadata.source,
          score: node.score,
          chunkIndex: node.metadata.chunkIndex,
        }));
      } else {
        // Vector-based search
        log("[Query] Using vector search mode");
        if (!lanceDb) {
          throw new Error("LanceDB not initialized");
        }

        log(`[Query] Querying vector store with topK=${config.topK}...`);
        const startVector = Date.now();
        const vectorResults = await lanceDb.query({
          indexName: actualTableName,
          tableName: actualTableName,
          queryVector: queryEmbedding,
          topK: config.topK,
          includeVector: false,
          includeAllColumns: true, // IMPORTANT: Include metadata fields
        });
        log(`[Query] Vector search completed in ${Date.now() - startVector}ms, found ${vectorResults.length} results`);

        results = vectorResults.map((result: any) => ({
          text: result.metadata?.text || result.metadata_text || result.text || "",
          source: result.metadata?.source || result.metadata_source || result.source || "",
          score: result.score || 0,
          chunkIndex: 0, // chunkIndex is not stored in vector DB, only in graph data
        }));
      }

      // Sort by score (highest first)
      results.sort((a, b) => b.score - a.score);

      log(`[Query] Returning ${results.length} results`);

      const returnValue = {
        results,
      };
      log("[Query] Return value:", JSON.stringify(returnValue, null, 2));
      return returnValue;
    } catch (error) {
      console.error("[Query] Error:", error);
      log("[Query] Error stack:", error instanceof Error ? error.stack : 'no stack');
      // Ensure we return valid structure even on error
      return {
        results: [],
      };
    }
  },
});

// Main async function to initialize and start the server
async function main() {
  log("[Init] Initializing MCP server...");

  // Initialize LanceDB
  log("[Init] Connecting to LanceDB...");
  const startLance = Date.now();
  lanceDb = await LanceVectorStore.create(lanceDbPath);
  log(`[Init] LanceDB connected in ${Date.now() - startLance}ms`);

  // Build GraphRAG once at startup if enabled and graph data exists
  if (config.enableGraph && hasGraphData) {
    log("[Init] Building GraphRAG instance (this may take several minutes for large indexes)...");
    const startGraph = Date.now();
    graphRag = graphStore.buildGraphRAG(
      config.dimensions,
      config.graphThreshold
    );
    if (graphRag) {
      log(`[Init] GraphRAG built in ${Date.now() - startGraph}ms`);
    } else {
      console.error("[Init] Failed to build GraphRAG, graph search will be disabled");
    }
  } else if (!config.enableGraph && hasGraphData) {
    log("[Init] Graph data available but --enable-graph not specified, using vector search only");
  }

  // Create MCP Server with dynamic name/description based on index
  const serverName = config.indexName 
    ? `${config.indexName} Query Server`
    : "Embeddings Query Server";
  
  const serverDescription = config.indexName
    ? `Query the '${config.indexName}' index using vector or graph-based search. This index contains embeddings created from source documents.`
    : "Query vector stores and knowledge graphs created by the embedder";

  const server = new MCPServer({
    name: serverName,
    version: "1.0.0",
    description: serverDescription,
    tools: {
      query_index: queryIndexTool,
    },
  });

  // Always show that server is ready (minimal output)
  console.info("MCP Server Ready");

  // Verbose details
  log("Index path:", config.indexPath);
  log("Table name:", actualTableName);
  log("Model:", config.model);
  log("Dimensions:", config.dimensions);
  log("Top K:", config.topK);
  log("Graph available:", hasGraphData && graphRag !== null);

  // Start the server
  await server.startStdio();
}

// Start the main function
main().catch((error) => {
  console.error("Failed to start MCP server:", error);
  process.exit(1);
});
