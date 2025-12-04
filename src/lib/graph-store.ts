import * as fs from "fs";
import * as path from "path";
import { GraphRAG } from "@mastra/rag";
import type { GraphChunkData } from "../types/index.js";

const GRAPH_DIR = "graph-data";
const CHUNKS_DIR = "chunks";
const EMBEDDINGS_DIR = "embeddings";
const CONFIG_FILE = "config.json";
const INDEX_FILE = "index.json";
const GRAPH_CACHE_FILE = "graph-structure.bin";
const BATCH_SIZE = 1000; // Chunks per batch file

// Graph build states
export enum GraphBuildState {
  IDLE = "idle",           // Not started
  BUILDING = "building",   // Currently building
  READY = "ready",         // Built and ready
  FAILED = "failed",       // Build failed
}

interface GraphConfig {
  version: string;
  dimension: number;
  threshold: number;
  createdAt: number;
  updatedAt: number;
}

interface GraphIndex {
  chunkCount: number;
  batchSize: number;
  lastUpdated: number;
}

interface ChunkBatch {
  chunks: GraphChunkData[];
}

interface GraphCacheMeta {
  version: string;
  dimension: number;
  threshold: number;
  chunkCount: number;
  createdAt: number;
}

/**
 * GraphStore handles loading of persisted GraphRAG data.
 * This is a read-only version adapted from the embedder project.
 * 
 * Now supports:
 * - Lazy loading (graph built on first query)
 * - Parallel batch loading
 * - Persistent graph caching
 * - State management for async graph building
 */
export class GraphStore {
  private graphDir: string;
  private chunksDir: string;
  private embeddingsDir: string;
  private configPath: string;
  private indexPath: string;
  private graphCachePath: string;

  private config: GraphConfig | null = null;
  private index: GraphIndex | null = null;

  // In-memory cache of chunks (loaded on demand)
  private chunksCache: Map<number, GraphChunkData[]> = new Map();
  private embeddingsCache: Map<number, number[][]> = new Map();

  // Graph state management
  private graphState: GraphBuildState = GraphBuildState.IDLE;
  private graphInstance: GraphRAG | null = null;
  private buildPromise: Promise<GraphRAG | null> | null = null;
  private buildError: Error | null = null;

  constructor(outputDir: string) {
    this.graphDir = path.join(outputDir, GRAPH_DIR);
    this.chunksDir = path.join(this.graphDir, CHUNKS_DIR);
    this.embeddingsDir = path.join(this.graphDir, EMBEDDINGS_DIR);
    this.configPath = path.join(this.graphDir, CONFIG_FILE);
    this.indexPath = path.join(this.graphDir, INDEX_FILE);
    this.graphCachePath = path.join(this.graphDir, GRAPH_CACHE_FILE);
  }

  /**
   * Load configuration and index if they exist
   */
  public load(): boolean {
    if (!fs.existsSync(this.graphDir)) {
      return false;
    }

    this.config = this.loadConfig();
    this.index = this.loadIndex();

    return this.config !== null && this.index !== null;
  }

  private loadConfig(): GraphConfig | null {
    if (!fs.existsSync(this.configPath)) {
      return null;
    }

    try {
      const content = fs.readFileSync(this.configPath, "utf-8");
      const config = JSON.parse(content) as GraphConfig;

      if (config.version !== "1.0") {
        console.warn(`Graph config version mismatch (${config.version})`);
        return null;
      }

      return config;
    } catch (error) {
      console.warn("Failed to load graph config");
      return null;
    }
  }

  private loadIndex(): GraphIndex | null {
    if (!fs.existsSync(this.indexPath)) {
      return null;
    }

    try {
      const content = fs.readFileSync(this.indexPath, "utf-8");
      return JSON.parse(content) as GraphIndex;
    } catch (error) {
      console.warn("Failed to load graph index");
      return null;
    }
  }

  private getChunkBatchPath(batchNum: number): string {
    return path.join(
      this.chunksDir,
      `batch-${batchNum.toString().padStart(4, "0")}.json`
    );
  }

  private getEmbeddingBatchPath(batchNum: number): string {
    return path.join(
      this.embeddingsDir,
      `batch-${batchNum.toString().padStart(4, "0")}.bin`
    );
  }

  private loadChunkBatch(batchNum: number): GraphChunkData[] {
    if (this.chunksCache.has(batchNum)) {
      return this.chunksCache.get(batchNum)!;
    }

    const batchPath = this.getChunkBatchPath(batchNum);
    if (!fs.existsSync(batchPath)) {
      return [];
    }

    try {
      const content = fs.readFileSync(batchPath, "utf-8");
      const batch = JSON.parse(content) as ChunkBatch;
      this.chunksCache.set(batchNum, batch.chunks);
      return batch.chunks;
    } catch (error) {
      console.warn(`Failed to load chunk batch ${batchNum}`);
      return [];
    }
  }

  private loadEmbeddingBatch(batchNum: number): number[][] {
    if (this.embeddingsCache.has(batchNum)) {
      return this.embeddingsCache.get(batchNum)!;
    }

    const batchPath = this.getEmbeddingBatchPath(batchNum);
    if (!fs.existsSync(batchPath)) {
      return [];
    }

    try {
      const buffer = fs.readFileSync(batchPath);
      if (!this.config) {
        throw new Error("Config not loaded");
      }
      const embeddings = this.decodeEmbeddings(buffer, this.config.dimension);
      this.embeddingsCache.set(batchNum, embeddings);
      return embeddings;
    } catch (error) {
      console.warn(`Failed to load embedding batch ${batchNum}`);
      return [];
    }
  }

  private decodeEmbeddings(buffer: Buffer, dimension: number): number[][] {
    let offset = 0;
    const count = buffer.readUInt32LE(offset);
    offset += 4;

    const embeddings: number[][] = [];
    for (let i = 0; i < count; i++) {
      const embedding: number[] = [];
      for (let j = 0; j < dimension; j++) {
        embedding.push(buffer.readFloatLE(offset));
        offset += 4;
      }
      embeddings.push(embedding);
    }

    return embeddings;
  }

  /**
   * Get statistics about the stored graph data
   */
  public getStats(): { nodeCount: number; updatedAt: number } | null {
    if (!this.index) {
      return null;
    }

    return {
      nodeCount: this.index.chunkCount,
      updatedAt: this.index.lastUpdated,
    };
  }

  /**
   * Check if there's existing graph data
   */
  public hasData(): boolean {
    return this.index !== null && this.index.chunkCount > 0;
  }

  /**
   * Get the stored configuration
   */
  public getConfig(): { dimension: number; threshold: number } | null {
    if (!this.config) {
      return null;
    }

    return {
      dimension: this.config.dimension,
      threshold: this.config.threshold,
    };
  }

  /**
   * Get the current graph build state
   */
  public getGraphState(): GraphBuildState {
    return this.graphState;
  }

  /**
   * Trigger background graph building using a worker thread
   * This is non-blocking and allows the server to start immediately
   * 
   * Note: Due to GraphRAG not being serializable, this mainly serves
   * to warm up file caches and test build performance
   */
  public buildGraphInBackground(dimension?: number, threshold?: number): void {
    if (this.graphState !== GraphBuildState.IDLE) {
      console.warn("[GraphStore] Graph already building or built, skipping background build");
      return;
    }

    if (!this.hasData() || !this.config || !this.index) {
      console.warn("[GraphStore] No graph data available for background build");
      return;
    }

    const dim = dimension ?? this.config.dimension;
    const thresh = threshold ?? this.config.threshold;

    console.error("[GraphStore] Starting background graph build in worker thread...");
    
    // Import worker_threads dynamically
    import("worker_threads").then(({ Worker }) => {
      const workerPath = new URL("./graph-worker.js", import.meta.url).pathname;
      
      const worker = new Worker(workerPath, {
        workerData: {
          indexPath: path.dirname(this.graphDir),
          dimension: dim,
          threshold: thresh,
        },
      });

      worker.on("message", (result: any) => {
        if (result.success) {
          console.error(`[GraphStore] Background graph build completed in ${result.buildTime}ms`);
        } else {
          console.error(`[GraphStore] Background graph build failed: ${result.error}`);
        }
      });

      worker.on("error", (error) => {
        console.error("[GraphStore] Worker thread error:", error);
      });

      worker.on("exit", (code) => {
        if (code !== 0) {
          console.error(`[GraphStore] Worker stopped with exit code ${code}`);
        }
      });
    }).catch((error) => {
      console.error("[GraphStore] Failed to start worker thread:", error);
    });
  }

  /**
   * Get the graph instance if ready, or trigger lazy building
   * This is the main entry point for getting a GraphRAG instance
   * 
   * @param dimension - Override dimension (uses stored if not provided)
   * @param threshold - Override threshold (uses stored if not provided)
   * @returns Promise resolving to GraphRAG instance or null if failed
   */
  public async getOrBuildGraphRAG(
    dimension?: number,
    threshold?: number
  ): Promise<GraphRAG | null> {
    // Return immediately if already ready
    if (this.graphState === GraphBuildState.READY && this.graphInstance) {
      return this.graphInstance;
    }

    // If already building, wait for that build to complete
    if (this.graphState === GraphBuildState.BUILDING && this.buildPromise) {
      return this.buildPromise;
    }

    // If failed previously, return null
    if (this.graphState === GraphBuildState.FAILED) {
      return null;
    }

    // Start building
    this.graphState = GraphBuildState.BUILDING;
    this.buildPromise = this.buildGraphRAGInternal(dimension, threshold);
    
    try {
      const result = await this.buildPromise;
      if (result) {
        this.graphState = GraphBuildState.READY;
        this.graphInstance = result;
      } else {
        this.graphState = GraphBuildState.FAILED;
      }
      return result;
    } catch (error) {
      this.graphState = GraphBuildState.FAILED;
      this.buildError = error instanceof Error ? error : new Error(String(error));
      console.error("[GraphStore] Graph build failed:", this.buildError);
      return null;
    }
  }

  /**
   * Internal method to build GraphRAG with parallel batch loading and caching
   */
  private async buildGraphRAGInternal(
    dimension?: number,
    threshold?: number
  ): Promise<GraphRAG | null> {
    if (!this.hasData() || !this.config || !this.index) {
      return null;
    }

    const dim = dimension ?? this.config.dimension;
    const thresh = threshold ?? this.config.threshold;

    if (dim === 0) {
      throw new Error("Graph dimension not set");
    }

    console.error(`[GraphStore] Creating GraphRAG with dimension=${dim}, threshold=${thresh}`);
    
    // Try to load from cache first
    const cachedGraph = await this.loadGraphFromCache(dim, thresh);
    if (cachedGraph) {
      console.error(`[GraphStore] Loaded graph from cache`);
      return cachedGraph;
    }

    // Build from scratch with parallel loading
    const graphRag = new GraphRAG(dim, thresh);
    const totalBatches = Math.ceil(this.index.chunkCount / BATCH_SIZE);

    console.error(`[GraphStore] Loading ${this.index.chunkCount} nodes from ${totalBatches} batches in parallel...`);
    const startLoad = Date.now();
    
    // Load all batches in parallel using Promise.all with progress tracking
    const batchNumbers = Array.from({ length: totalBatches }, (_, i) => i);
    let completedBatches = 0;
    
    const batchResults = await Promise.all(
      batchNumbers.map(async (batchNum) => {
        const chunks = this.loadChunkBatch(batchNum);
        const embeddings = this.loadEmbeddingBatch(batchNum);
        
        // Log progress for large datasets
        completedBatches++;
        if (totalBatches > 10 && completedBatches % Math.ceil(totalBatches / 10) === 0) {
          const percent = Math.round((completedBatches / totalBatches) * 100);
          console.error(`[GraphStore] Loading progress: ${percent}% (${completedBatches}/${totalBatches} batches)`);
        }
        
        return { chunks, embeddings };
      })
    );

    // Flatten results
    const allChunks: GraphChunkData[] = [];
    const allEmbeddings: number[][] = [];
    for (const { chunks, embeddings } of batchResults) {
      allChunks.push(...chunks);
      allEmbeddings.push(...embeddings);
    }
    
    console.error(`[GraphStore] Loaded ${allChunks.length} chunks and ${allEmbeddings.length} embeddings in ${Date.now() - startLoad}ms`);

    // Convert to GraphRAG format
    console.error(`[GraphStore] Converting to GraphRAG format...`);
    const startConvert = Date.now();
    const graphChunks = allChunks.map((chunk) => ({
      text: chunk.text,
      metadata: {
        id: chunk.id,
        source: chunk.source,
        chunkIndex: chunk.chunkIndex,
      },
    }));

    const graphEmbeddings = allEmbeddings.map((emb) => ({
      vector: emb,
    }));
    console.error(`[GraphStore] Conversion completed in ${Date.now() - startConvert}ms`);

    // Build the graph structure
    console.error(`[GraphStore] Building graph structure (this may take a while)...`);
    const startBuild = Date.now();
    graphRag.createGraph(graphChunks, graphEmbeddings);
    const buildTime = Date.now() - startBuild;
    console.error(`[GraphStore] Graph built in ${buildTime}ms`);

    // Cache the built graph asynchronously (don't wait for it)
    this.saveGraphToCache(graphRag, dim, thresh, allChunks.length).catch((error) => {
      console.error("[GraphStore] Failed to save graph cache:", error);
    });

    return graphRag;
  }

  /**
   * Load a previously cached graph structure from disk
   */
  private async loadGraphFromCache(
    dimension: number,
    threshold: number
  ): Promise<GraphRAG | null> {
    if (!fs.existsSync(this.graphCachePath)) {
      return null;
    }

    try {
      const cacheMetaPath = this.graphCachePath + ".meta.json";
      if (!fs.existsSync(cacheMetaPath)) {
        return null;
      }

      // Check if cache is valid
      const metaContent = fs.readFileSync(cacheMetaPath, "utf-8");
      const meta = JSON.parse(metaContent) as GraphCacheMeta;

      // Validate cache matches current configuration
      if (
        meta.version !== "1.0" ||
        meta.dimension !== dimension ||
        meta.threshold !== threshold ||
        meta.chunkCount !== this.index?.chunkCount
      ) {
        console.error("[GraphStore] Graph cache is stale, rebuilding...");
        return null;
      }

      // Check if cache is newer than data files
      if (this.index && meta.createdAt < this.index.lastUpdated) {
        console.error("[GraphStore] Graph cache is older than data, rebuilding...");
        return null;
      }

      // Load the graph (note: GraphRAG doesn't have built-in serialization)
      // We'll need to rebuild for now, but cache the chunks/embeddings
      // This is a limitation of the Mastra GraphRAG library
      console.error("[GraphStore] Cache found but GraphRAG doesn't support deserialization");
      return null;
    } catch (error) {
      console.error("[GraphStore] Failed to load graph cache:", error);
      return null;
    }
  }

  /**
   * Save the built graph structure to disk for future use
   */
  private async saveGraphToCache(
    graphRag: GraphRAG,
    dimension: number,
    threshold: number,
    chunkCount: number
  ): Promise<void> {
    try {
      // Create metadata
      const meta: GraphCacheMeta = {
        version: "1.0",
        dimension,
        threshold,
        chunkCount,
        createdAt: Date.now(),
      };

      const cacheMetaPath = this.graphCachePath + ".meta.json";
      fs.writeFileSync(cacheMetaPath, JSON.stringify(meta, null, 2));

      // Note: GraphRAG from Mastra doesn't expose serialization methods
      // We save metadata for now, and in the future could serialize if API allows
      console.error("[GraphStore] Graph cache metadata saved (full serialization not supported by Mastra GraphRAG)");
    } catch (error) {
      console.error("[GraphStore] Failed to save graph cache:", error);
      throw error;
    }
  }

  /**
   * Get all chunks (loads from all batches)
   */
  public getChunks(): GraphChunkData[] {
    if (!this.index) {
      return [];
    }

    const totalBatches = Math.ceil(this.index.chunkCount / BATCH_SIZE);
    const allChunks: GraphChunkData[] = [];

    for (let batchNum = 0; batchNum < totalBatches; batchNum++) {
      const chunks = this.loadChunkBatch(batchNum);
      allChunks.push(...chunks);
    }

    return allChunks;
  }

  /**
   * Get all embeddings (loads from all batches)
   */
  public getEmbeddings(): number[][] {
    if (!this.index) {
      return [];
    }

    const totalBatches = Math.ceil(this.index.chunkCount / BATCH_SIZE);
    const allEmbeddings: number[][] = [];

    for (let batchNum = 0; batchNum < totalBatches; batchNum++) {
      const embeddings = this.loadEmbeddingBatch(batchNum);
      allEmbeddings.push(...embeddings);
    }

    return allEmbeddings;
  }
}
