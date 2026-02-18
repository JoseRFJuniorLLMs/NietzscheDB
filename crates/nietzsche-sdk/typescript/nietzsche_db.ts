/**
 * NietzscheDB TypeScript/Node.js gRPC client SDK.
 *
 * Generated from `proto/nietzsche.proto`.  Requires the `@grpc/grpc-js` and
 * `@grpc/proto-loader` packages.
 *
 * ## Installation
 * ```bash
 * npm install @grpc/grpc-js @grpc/proto-loader
 * # TypeScript types
 * npm install -D ts-proto
 * ```
 *
 * ## Regenerate stubs with ts-proto
 * ```bash
 * npx protoc \
 *   --plugin=./node_modules/.bin/protoc-gen-ts_proto \
 *   --ts_proto_out=. \
 *   --ts_proto_opt=outputServices=grpc-js \
 *   -I../../crates/nietzsche-api/proto \
 *   nietzsche.proto
 * ```
 *
 * ## Example
 * ```typescript
 * import { NietzscheClient } from './nietzsche_db';
 *
 * const client = new NietzscheClient('localhost:50051');
 *
 * const node = await client.insertNode({
 *   coords: [0.1, 0.2, 0.3],
 *   content: { text: 'Thus spoke Zarathustra' },
 *   nodeType: 'Episodic',
 * });
 * console.log('inserted:', node.id);
 *
 * const result = await client.query('MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 10');
 * console.log(`${result.nodes.length} nodes returned`);
 *
 * const report = await client.triggerSleep();
 * console.log(`committed=${report.committed}, ΔH=${report.hausdorfDelta.toFixed(4)}`);
 *
 * client.close();
 * ```
 */

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import * as path from 'path';

// ---------------------------------------------------------------------------
// Proto loading
// ---------------------------------------------------------------------------

const PROTO_PATH = path.resolve(__dirname, '../../nietzsche-api/proto/nietzsche.proto');

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const proto = grpc.loadPackageDefinition(packageDefinition) as any;
const NietzscheDBStub = proto.nietzsche.NietzscheDB;

// ---------------------------------------------------------------------------
// TypeScript interfaces (mirrors proto messages)
// ---------------------------------------------------------------------------

export interface PoincareVector {
  coords: number[];
  dim:    number;
}

export interface NodeResponse {
  found:           boolean;
  id:              string;
  embedding?:      PoincareVector;
  energy:          number;
  depth:           number;
  hausdorffLocal:  number;
  createdAt:       number;
  content:         Uint8Array;
  nodeType:        string;
}

export interface StatusResponse {
  status: string;
  error:  string;
}

export interface EdgeResponse {
  success: boolean;
  id:      string;
}

export interface KnnResult {
  id:       string;
  distance: number;
}

export interface KnnResponse {
  results: KnnResult[];
}

export interface NodePair {
  from?: NodeResponse;
  to?:   NodeResponse;
}

export interface QueryResponse {
  nodes:     NodeResponse[];
  nodePairs: NodePair[];
  pathIds:   string[];
  error:     string;
}

export interface TraversalResponse {
  visitedIds: string[];
  costs:      number[];
}

export interface DiffusionScale {
  t:       number;
  nodeIds: string[];
  scores:  number[];
}

export interface DiffusionResponse {
  scales: DiffusionScale[];
}

export interface SleepResponse {
  hausdorffBefore:  number;
  hausdorffAfter:   number;
  hausdorffDelta:   number;
  committed:        boolean;
  nodesPerturbed:   number;
  snapshotNodes:    number;
}

export interface StatsResponse {
  nodeCount: number;
  edgeCount: number;
  version:   string;
}

// ---------------------------------------------------------------------------
// Parameter interfaces
// ---------------------------------------------------------------------------

export interface InsertNodeParams {
  coords:    number[];
  content?:  Record<string, unknown>;
  nodeType?: string;
  energy?:   number;
  id?:       string;
}

export interface InsertEdgeParams {
  from:      string;
  to:        string;
  edgeType?: string;
  weight?:   number;
  id?:       string;
}

export interface SleepParams {
  noise?:               number;
  adamSteps?:           number;
  adamLr?:              number;
  hausdorffThreshold?:  number;
  rngSeed?:             number;
}

export interface TraversalParams {
  maxDepth?:  number;
  maxNodes?:  number;
  maxCost?:   number;
  energyMin?: number;
}

// ---------------------------------------------------------------------------
// NietzscheClient
// ---------------------------------------------------------------------------

/**
 * Async gRPC client for NietzscheDB.
 *
 * All methods return Promises and can be used with async/await.
 */
export class NietzscheClient {
  private stub: any;

  /**
   * @param address  Server address, e.g. `"localhost:50051"`.
   * @param secure   Use TLS if `true` (default `false`).
   */
  constructor(address: string, secure = false) {
    const credentials = secure
      ? grpc.credentials.createSsl()
      : grpc.credentials.createInsecure();
    this.stub = new NietzscheDBStub(address, credentials);
  }

  /** Close the gRPC channel. */
  close(): void {
    grpc.closeClient(this.stub);
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  private call<Req, Res>(method: string, req: Req): Promise<Res> {
    return new Promise((resolve, reject) => {
      this.stub[method](req, (err: grpc.ServiceError | null, res: Res) => {
        if (err) reject(err); else resolve(res);
      });
    });
  }

  // ── Node CRUD ────────────────────────────────────────────────────────────

  async insertNode(params: InsertNodeParams): Promise<NodeResponse> {
    const contentBytes = params.content
      ? Buffer.from(JSON.stringify(params.content))
      : Buffer.alloc(0);

    return this.call('InsertNode', {
      id:        params.id ?? '',
      embedding: { coords: params.coords, dim: params.coords.length },
      content:   contentBytes,
      node_type: params.nodeType ?? 'Semantic',
      energy:    params.energy   ?? 1.0,
    });
  }

  async getNode(id: string): Promise<NodeResponse> {
    return this.call('GetNode', { id });
  }

  async deleteNode(id: string): Promise<StatusResponse> {
    return this.call('DeleteNode', { id });
  }

  async updateEnergy(nodeId: string, energy: number): Promise<StatusResponse> {
    return this.call('UpdateEnergy', { node_id: nodeId, energy });
  }

  // ── Edge CRUD ────────────────────────────────────────────────────────────

  async insertEdge(params: InsertEdgeParams): Promise<EdgeResponse> {
    return this.call('InsertEdge', {
      id:        params.id       ?? '',
      from:      params.from,
      to:        params.to,
      edge_type: params.edgeType ?? 'Association',
      weight:    params.weight   ?? 1.0,
    });
  }

  async deleteEdge(id: string): Promise<StatusResponse> {
    return this.call('DeleteEdge', { id });
  }

  // ── NQL query ────────────────────────────────────────────────────────────

  /**
   * Execute a Nietzsche Query Language statement.
   *
   * @example
   * ```typescript
   * const r = await client.query(
   *   "MATCH (n:Memory) WHERE n.energy > 0.3 RETURN n ORDER BY n.energy DESC LIMIT 10"
   * );
   * ```
   */
  async query(nql: string): Promise<QueryResponse> {
    return this.call('Query', { nql });
  }

  // ── Vector KNN ───────────────────────────────────────────────────────────

  async knnSearch(coords: number[], k: number = 10): Promise<KnnResponse> {
    return this.call('KnnSearch', { query_coords: coords, k });
  }

  // ── Traversal ────────────────────────────────────────────────────────────

  async bfs(startId: string, params: TraversalParams = {}): Promise<TraversalResponse> {
    return this.call('Bfs', {
      start_node_id: startId,
      max_depth:     params.maxDepth  ?? 10,
      max_nodes:     params.maxNodes  ?? 1000,
      max_cost:      params.maxCost   ?? 0,
      energy_min:    params.energyMin ?? 0,
    });
  }

  async dijkstra(startId: string, params: TraversalParams = {}): Promise<TraversalResponse> {
    return this.call('Dijkstra', {
      start_node_id: startId,
      max_depth:     0,
      max_nodes:     params.maxNodes  ?? 1000,
      max_cost:      params.maxCost   ?? 0,
      energy_min:    params.energyMin ?? 0,
    });
  }

  // ── Diffusion ────────────────────────────────────────────────────────────

  async diffuse(
    sourceIds:  string[],
    tValues:    number[] = [0.1, 1.0, 10.0],
    kChebyshev: number  = 10,
  ): Promise<DiffusionResponse> {
    return this.call('Diffuse', {
      source_ids:  sourceIds,
      t_values:    tValues,
      k_chebyshev: kChebyshev,
    });
  }

  // ── Sleep cycle ──────────────────────────────────────────────────────────

  async triggerSleep(params: SleepParams = {}): Promise<SleepResponse> {
    return this.call('TriggerSleep', {
      noise:                params.noise               ?? 0.02,
      adam_steps:           params.adamSteps           ?? 10,
      adam_lr:              params.adamLr              ?? 0.005,
      hausdorff_threshold:  params.hausdorffThreshold  ?? 0.15,
      rng_seed:             params.rngSeed             ?? 0,
    });
  }

  // ── Admin ────────────────────────────────────────────────────────────────

  async getStats(): Promise<StatsResponse> {
    return this.call('GetStats', {});
  }

  async healthCheck(): Promise<StatusResponse> {
    return this.call('HealthCheck', {});
  }
}
