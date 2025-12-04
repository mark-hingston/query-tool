/**
 * Worker thread for building GraphRAG instances in the background
 * This allows the main thread to remain responsive while building large graphs
 */

import { parentPort, workerData } from "worker_threads";
import { GraphRAG } from "@mastra/rag";
import * as fs from "fs";
import * as path from "path";

interface WorkerData {
  indexPath: string;
  dimension: number;
  threshold: number;
}

interface WorkerResult {
  success: boolean;
  error?: string;
  buildTime?: number;
}

interface ChunkBatch {
  chunks: Array<{
    id: string;
    text: string;
    source: string;
    chunkIndex: number;
  }>;
}

const GRAPH_DIR = "graph-data";
const CHUNKS_DIR = "chunks";
const EMBEDDINGS_DIR = "embeddings";
const BATCH_SIZE = 1000;

function decodeEmbeddings(buffer: Buffer, dimension: number): number[][] {
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

async function buildGraph(data: WorkerData): Promise<WorkerResult> {
  const startTime = Date.now();

  try {
    const graphDir = path.join(data.indexPath, GRAPH_DIR);
    const chunksDir = path.join(graphDir, CHUNKS_DIR);
    const embeddingsDir = path.join(graphDir, EMBEDDINGS_DIR);
    const indexPath = path.join(graphDir, "index.json");

    // Load index
    const indexContent = fs.readFileSync(indexPath, "utf-8");
    const index = JSON.parse(indexContent) as { chunkCount: number };
    const totalBatches = Math.ceil(index.chunkCount / BATCH_SIZE);

    console.error(`[GraphWorker] Loading ${index.chunkCount} nodes from ${totalBatches} batches...`);

    // Load all batches in parallel
    const batchNumbers = Array.from({ length: totalBatches }, (_, i) => i);
    const batchResults = await Promise.all(
      batchNumbers.map(async (batchNum) => {
        const chunkPath = path.join(
          chunksDir,
          `batch-${batchNum.toString().padStart(4, "0")}.json`
        );
        const embeddingPath = path.join(
          embeddingsDir,
          `batch-${batchNum.toString().padStart(4, "0")}.bin`
        );

        const chunkContent = fs.readFileSync(chunkPath, "utf-8");
        const chunkBatch = JSON.parse(chunkContent) as ChunkBatch;

        const embeddingBuffer = fs.readFileSync(embeddingPath);
        const embeddings = decodeEmbeddings(embeddingBuffer, data.dimension);

        return { chunks: chunkBatch.chunks, embeddings };
      })
    );

    // Flatten results
    const allChunks = batchResults.flatMap((r) => r.chunks);
    const allEmbeddings = batchResults.flatMap((r) => r.embeddings);

    console.error(`[GraphWorker] Loaded ${allChunks.length} chunks and ${allEmbeddings.length} embeddings`);

    // Build graph
    console.error(`[GraphWorker] Building graph structure...`);
    const graphRag = new GraphRAG(data.dimension, data.threshold);

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

    graphRag.createGraph(graphChunks, graphEmbeddings);

    const buildTime = Date.now() - startTime;
    console.error(`[GraphWorker] Graph built successfully in ${buildTime}ms`);

    // Note: We can't transfer the GraphRAG instance back to main thread
    // This worker is mainly useful for warming up the cache and testing
    // In production, we'd need to serialize the graph structure

    return {
      success: true,
      buildTime,
    };
  } catch (error) {
    console.error("[GraphWorker] Error building graph:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// Main worker execution
if (parentPort) {
  const data = workerData as WorkerData;
  buildGraph(data)
    .then((result) => {
      parentPort!.postMessage(result);
    })
    .catch((error) => {
      parentPort!.postMessage({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    });
}
