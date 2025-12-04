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
import { GraphStore, GraphBuildState } from "./lib/graph-store.js";

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
  .option("--dimensions <number>", "Embedding dimensions", "1024")
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
    "--preload-graph",
    "Start building graph in background at startup (still allows immediate queries via vector search)",
    false
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
  preloadGraph: options.preloadGraph || false,
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

// Load graph store (if it exists) - but don't build the graph yet (lazy loading)
const graphStore = new GraphStore(config.indexPath);
const hasGraphData = graphStore.load() && graphStore.hasData();

if (hasGraphData) {
  log("Graph data found and loaded");
  const stats = graphStore.getStats();
  if (stats) {
    log(`Graph contains ${stats.nodeCount} nodes`);
  }
  if (config.enableGraph) {
    log("Graph search enabled - graph will be built on first query (lazy loading)");
  } else {
    log("Graph search disabled - use --enable-graph to enable");
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
      let graphInstance: any | null = null;

      if (mode === "graph") {
        if (!config.enableGraph || !hasGraphData) {
          throw new Error("Graph search requested but graph is not available. Use --enable-graph and ensure graph data exists.");
        }
        useGraph = true;
      } else if (mode === "auto") {
        // Only use graph if enabled and data exists
        useGraph = config.enableGraph && hasGraphData;
      }

      // If we want to use graph, try to get/build it
      if (useGraph) {
        const graphState = graphStore.getGraphState();
        log(`[Query] Graph state: ${graphState}`);

        if (graphState === GraphBuildState.BUILDING) {
          console.warn("[Query] WARNING: Graph is still building. Falling back to vector search for this query.");
          log("[Query] To avoid this warning, wait for graph to finish building before querying.");
          useGraph = false;
        } else if (graphState === GraphBuildState.FAILED) {
          console.warn("[Query] WARNING: Graph build failed previously. Falling back to vector search.");
          useGraph = false;
        } else {
          // Try to get or build the graph (lazy loading)
          log("[Query] Getting or building graph instance (lazy loading)...");
          const startGetGraph = Date.now();
          graphInstance = await graphStore.getOrBuildGraphRAG(
            config.dimensions,
            config.graphThreshold
          );
          log(`[Query] Graph ready in ${Date.now() - startGetGraph}ms`);

          if (!graphInstance) {
            console.warn("[Query] WARNING: Failed to get graph instance. Falling back to vector search.");
            useGraph = false;
          }
        }
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

      if (useGraph && graphInstance) {
        // Graph-based search
        log("[Query] Using graph search mode");

        log(`[Query] Querying graph with topK=${config.topK}, randomWalkSteps=${config.randomWalkSteps}...`);
        const startGraph = Date.now();
        const graphResults = await graphInstance.query({
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

  // Note: GraphRAG is now built lazily on first query (if enabled)
  // This dramatically improves startup time for large indexes
  if (config.enableGraph && hasGraphData) {
    log("[Init] Graph search enabled - graph will be built on first query (lazy loading)");
    
    // Optionally start building in the background
    if (config.preloadGraph) {
      log("[Init] Starting graph preload in background...");
      // Trigger lazy loading asynchronously without blocking startup
      graphStore.getOrBuildGraphRAG(config.dimensions, config.graphThreshold).then(() => {
        log("[Init] Graph preload completed");
      }).catch((error) => {
        console.error("[Init] Graph preload failed:", error);
      });
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
  log("Graph available:", hasGraphData && config.enableGraph);

  // Start the server
  await server.startStdio();
}

// Start the main function
main().catch((error) => {
  console.error("Failed to start MCP server:", error);
  process.exit(1);
});
