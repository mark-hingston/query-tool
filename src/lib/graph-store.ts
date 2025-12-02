import * as fs from "fs";
import * as path from "path";
import { GraphRAG } from "@mastra/rag";
import type { GraphChunkData } from "../types/index.js";

const GRAPH_DIR = "graph-data";
const CHUNKS_DIR = "chunks";
const EMBEDDINGS_DIR = "embeddings";
const CONFIG_FILE = "config.json";
const INDEX_FILE = "index.json";
const BATCH_SIZE = 1000; // Chunks per batch file

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

/**
 * GraphStore handles loading of persisted GraphRAG data.
 * This is a read-only version adapted from the embedder project.
 */
export class GraphStore {
  private graphDir: string;
  private chunksDir: string;
  private embeddingsDir: string;
  private configPath: string;
  private indexPath: string;

  private config: GraphConfig | null = null;
  private index: GraphIndex | null = null;

  // In-memory cache of chunks (loaded on demand)
  private chunksCache: Map<number, GraphChunkData[]> = new Map();
  private embeddingsCache: Map<number, number[][]> = new Map();

  constructor(outputDir: string) {
    this.graphDir = path.join(outputDir, GRAPH_DIR);
    this.chunksDir = path.join(this.graphDir, CHUNKS_DIR);
    this.embeddingsDir = path.join(this.graphDir, EMBEDDINGS_DIR);
    this.configPath = path.join(this.graphDir, CONFIG_FILE);
    this.indexPath = path.join(this.graphDir, INDEX_FILE);
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

  private getBatchNumber(globalIndex: number): number {
    return Math.floor(globalIndex / BATCH_SIZE);
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
   * Build a GraphRAG instance from the persisted data
   *
   * @param dimension - Override dimension (uses stored if not provided)
   * @param threshold - Override threshold (uses stored if not provided)
   */
  public buildGraphRAG(
    dimension?: number,
    threshold?: number
  ): GraphRAG | null {
    if (!this.hasData() || !this.config || !this.index) {
      return null;
    }

    const dim = dimension ?? this.config.dimension;
    const thresh = threshold ?? this.config.threshold;

    if (dim === 0) {
      throw new Error("Graph dimension not set");
    }

    const graphRag = new GraphRAG(dim, thresh);

    // Load all chunks and embeddings from batches
    const totalBatches = Math.ceil(this.index.chunkCount / BATCH_SIZE);
    const allChunks: GraphChunkData[] = [];
    const allEmbeddings: number[][] = [];

    for (let batchNum = 0; batchNum < totalBatches; batchNum++) {
      const chunks = this.loadChunkBatch(batchNum);
      const embeddings = this.loadEmbeddingBatch(batchNum);
      allChunks.push(...chunks);
      allEmbeddings.push(...embeddings);
    }

    // Convert to GraphRAG format
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

    // Build the graph
    graphRag.createGraph(graphChunks, graphEmbeddings);

    return graphRag;
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
