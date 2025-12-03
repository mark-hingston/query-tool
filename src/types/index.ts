// Type definitions copied from embedder project

export interface GraphChunkData {
  id: string;
  text: string;
  source: string;
  chunkIndex: number;
}

export interface GraphEmbeddingData {
  id: string;
  embedding: number[];
}

export interface PersistedGraphData {
  version: string;
  createdAt: number;
  updatedAt: number;
  config: {
    dimension: number;
    threshold: number;
  };
  chunks: GraphChunkData[];
  embeddings: GraphEmbeddingData[];
}

// CLI configuration types
export interface ServerConfig {
  // Required
  indexPath: string;
  baseUrl: string;
  model: string;

  // Optional - Index Settings
  indexName?: string;
  tableName: string;
  dimensions: number;

  // Optional - Query Settings
  topK: number;

  // Optional - Graph Settings
  enableGraph: boolean;
  graphThreshold: number;
  randomWalkSteps: number;
  restartProb: number;

  // Optional - Logging Settings
  verbose: boolean;
}

// Query tool input
export interface QueryInput {
  query: string;
  mode: "auto" | "vector" | "graph";
}

// Query tool output
export interface QueryResult {
  text: string;
  source: string;
  score: number;
  chunkIndex: number;
}
