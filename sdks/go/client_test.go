// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"encoding/json"
	"net"
	"testing"

	pb "nietzsche-sdk/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

const bufSize = 1024 * 1024

// mockServer implements the NietzscheDB gRPC service for testing.
type mockServer struct {
	pb.UnimplementedNietzscheDBServer
}

func (s *mockServer) HealthCheck(_ context.Context, _ *pb.Empty) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) GetStats(_ context.Context, _ *pb.Empty) (*pb.StatsResponse, error) {
	return &pb.StatsResponse{
		NodeCount:    42,
		EdgeCount:    17,
		Version:      "0.12.0-test",
		SensoryCount: 5,
	}, nil
}

func (s *mockServer) InsertNode(_ context.Context, req *pb.InsertNodeRequest) (*pb.NodeResponse, error) {
	id := req.Id
	if id == "" {
		id = "auto-generated-uuid"
	}
	return &pb.NodeResponse{
		Found:     true,
		Id:        id,
		Embedding: req.Embedding,
		Energy:    req.Energy,
		Content:   req.Content,
		NodeType:  req.NodeType,
		CreatedAt: 1708000000,
	}, nil
}

func (s *mockServer) GetNode(_ context.Context, req *pb.NodeIdRequest) (*pb.NodeResponse, error) {
	if req.Id == "not-found" {
		return &pb.NodeResponse{Found: false}, nil
	}
	content, _ := json.Marshal(map[string]string{"text": "hello from test"})
	return &pb.NodeResponse{
		Found:    true,
		Id:       req.Id,
		Energy:   0.85,
		Depth:    0.3,
		NodeType: "Semantic",
		Content:  content,
		Embedding: &pb.PoincareVector{
			Coords: []float64{0.1, 0.2, 0.3},
			Dim:    3,
		},
		CreatedAt: 1708000000,
	}, nil
}

func (s *mockServer) DeleteNode(_ context.Context, _ *pb.NodeIdRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) UpdateEnergy(_ context.Context, _ *pb.UpdateEnergyRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) InsertEdge(_ context.Context, req *pb.InsertEdgeRequest) (*pb.EdgeResponse, error) {
	id := req.Id
	if id == "" {
		id = "edge-auto-uuid"
	}
	return &pb.EdgeResponse{Success: true, Id: id}, nil
}

func (s *mockServer) DeleteEdge(_ context.Context, _ *pb.EdgeIdRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) KnnSearch(_ context.Context, req *pb.KnnRequest) (*pb.KnnResponse, error) {
	results := make([]*pb.KnnResult, 0, req.K)
	for i := uint32(0); i < req.K && i < 3; i++ {
		results = append(results, &pb.KnnResult{
			Id:       "knn-node-" + string(rune('a'+i)),
			Distance: float64(i) * 0.5,
		})
	}
	return &pb.KnnResponse{Results: results}, nil
}

func (s *mockServer) Query(_ context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	content, _ := json.Marshal(map[string]string{"matched": "true"})
	return &pb.QueryResponse{
		Nodes: []*pb.NodeResponse{
			{Found: true, Id: "query-result-1", Content: content, NodeType: "Semantic"},
		},
		Explain: "SCAN default → filter → 1 node",
	}, nil
}

func (s *mockServer) Bfs(_ context.Context, _ *pb.TraversalRequest) (*pb.TraversalResponse, error) {
	return &pb.TraversalResponse{
		VisitedIds: []string{"node-a", "node-b", "node-c"},
	}, nil
}

func (s *mockServer) Dijkstra(_ context.Context, _ *pb.TraversalRequest) (*pb.TraversalResponse, error) {
	return &pb.TraversalResponse{
		VisitedIds: []string{"node-a", "node-b"},
		Costs:      []float64{0.0, 1.5},
	}, nil
}

func (s *mockServer) Diffuse(_ context.Context, req *pb.DiffusionRequest) (*pb.DiffusionResponse, error) {
	scales := make([]*pb.DiffusionScale, len(req.TValues))
	for i, t := range req.TValues {
		scales[i] = &pb.DiffusionScale{
			T:       t,
			NodeIds: []string{"diffuse-1", "diffuse-2"},
			Scores:  []float64{0.9, 0.4},
		}
	}
	return &pb.DiffusionResponse{Scales: scales}, nil
}

func (s *mockServer) CreateCollection(_ context.Context, req *pb.CreateCollectionRequest) (*pb.CreateCollectionResponse, error) {
	return &pb.CreateCollectionResponse{Created: true, Collection: req.Collection}, nil
}

func (s *mockServer) DropCollection(_ context.Context, _ *pb.DropCollectionRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) ListCollections(_ context.Context, _ *pb.Empty) (*pb.ListCollectionsResponse, error) {
	return &pb.ListCollectionsResponse{
		Collections: []*pb.CollectionInfoProto{
			{Collection: "default", Dim: 3072, Metric: "cosine", NodeCount: 100, EdgeCount: 50},
			{Collection: "eva_memory", Dim: 128, Metric: "poincare", NodeCount: 10, EdgeCount: 3},
		},
	}, nil
}

func (s *mockServer) TriggerSleep(_ context.Context, _ *pb.SleepRequest) (*pb.SleepResponse, error) {
	return &pb.SleepResponse{
		HausdorffBefore: 0.42,
		HausdorffAfter:  0.38,
		HausdorffDelta:  -0.04,
		Committed:       true,
		NodesPerturbed:  15,
		SnapshotNodes:   100,
	}, nil
}

func (s *mockServer) InvokeZaratustra(_ context.Context, _ *pb.ZaratustraRequest) (*pb.ZaratustraResponse, error) {
	return &pb.ZaratustraResponse{
		NodesUpdated:      50,
		MeanEnergyBefore:  0.6,
		MeanEnergyAfter:   0.72,
		TotalEnergyDelta:  6.0,
		EchoesCreated:     3,
		EchoesEvicted:     1,
		TotalEchoes:       12,
		EliteCount:        5,
		EliteThreshold:    0.9,
		MeanEliteEnergy:   0.95,
		MeanBaseEnergy:    0.65,
		EliteNodeIds:      []string{"elite-1", "elite-2"},
		DurationMs:        150,
		CyclesRun:         1,
	}, nil
}

func (s *mockServer) InsertSensory(_ context.Context, _ *pb.InsertSensoryRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

func (s *mockServer) GetSensory(_ context.Context, _ *pb.NodeIdRequest) (*pb.SensoryResponse, error) {
	return &pb.SensoryResponse{
		Found:                 true,
		NodeId:                "sensory-node",
		Modality:              "audio",
		Dim:                   256,
		QuantLevel:            "f32",
		ReconstructionQuality: 0.98,
		CompressionRatio:      0.15,
		EncoderVersion:        1,
		ByteSize:              1024,
	}, nil
}

func (s *mockServer) Reconstruct(_ context.Context, req *pb.ReconstructRequest) (*pb.ReconstructResponse, error) {
	return &pb.ReconstructResponse{
		Found:         true,
		NodeId:        req.NodeId,
		Latent:        []float32{0.1, 0.2, 0.3},
		Modality:      "audio",
		Quality:       0.95,
		OriginalShape: []byte(`{"channels":1,"sample_rate":16000}`),
	}, nil
}

func (s *mockServer) DegradeSensory(_ context.Context, _ *pb.NodeIdRequest) (*pb.StatusResponse, error) {
	return &pb.StatusResponse{Status: "ok"}, nil
}

// ── Test helpers ────────────────────────────────────────────────────────────

func startMockServer(t *testing.T) *NietzscheClient {
	t.Helper()

	lis := bufconn.Listen(bufSize)
	srv := grpc.NewServer()
	pb.RegisterNietzscheDBServer(srv, &mockServer{})

	go func() {
		if err := srv.Serve(lis); err != nil {
			t.Logf("mock server exited: %v", err)
		}
	}()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(
		"passthrough:///bufconn",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
	)
	if err != nil {
		t.Fatalf("failed to dial bufconn: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	return &NietzscheClient{
		conn: conn,
		stub: pb.NewNietzscheDBClient(conn),
	}
}

// ── Tests ───────────────────────────────────────────────────────────────────

func TestHealthCheck(t *testing.T) {
	client := startMockServer(t)
	if err := client.HealthCheck(context.Background()); err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
}

func TestGetStats(t *testing.T) {
	client := startMockServer(t)
	stats, err := client.GetStats(context.Background())
	if err != nil {
		t.Fatalf("GetStats failed: %v", err)
	}
	if stats.NodeCount != 42 {
		t.Errorf("expected NodeCount=42, got %d", stats.NodeCount)
	}
	if stats.Version != "0.12.0-test" {
		t.Errorf("expected Version=0.12.0-test, got %s", stats.Version)
	}
}

func TestInsertAndGetNode(t *testing.T) {
	client := startMockServer(t)
	ctx := context.Background()

	// Insert
	node, err := client.InsertNode(ctx, InsertNodeOpts{
		ID:         "test-node-1",
		Coords:     []float64{0.1, 0.2, 0.3},
		Content:    map[string]string{"text": "hello"},
		NodeType:   "Semantic",
		Energy:     0.9,
		Collection: "default",
	})
	if err != nil {
		t.Fatalf("InsertNode failed: %v", err)
	}
	if node.ID != "test-node-1" {
		t.Errorf("expected ID=test-node-1, got %s", node.ID)
	}

	// Get
	got, err := client.GetNode(ctx, "test-node-1", "default")
	if err != nil {
		t.Fatalf("GetNode failed: %v", err)
	}
	if !got.Found {
		t.Error("expected Found=true")
	}
	if got.Content["text"] != "hello from test" {
		t.Errorf("unexpected content: %v", got.Content)
	}
	if len(got.Embedding) != 3 {
		t.Errorf("expected 3-d embedding, got %d", len(got.Embedding))
	}
}

func TestInsertNodeAutoID(t *testing.T) {
	client := startMockServer(t)
	node, err := client.InsertNode(context.Background(), InsertNodeOpts{
		Coords:  []float64{0.5, 0.5},
		Content: "auto-id test",
	})
	if err != nil {
		t.Fatalf("InsertNode failed: %v", err)
	}
	if node.ID != "auto-generated-uuid" {
		t.Errorf("expected auto-generated ID, got %s", node.ID)
	}
}

func TestGetNodeNotFound(t *testing.T) {
	client := startMockServer(t)
	got, err := client.GetNode(context.Background(), "not-found", "")
	if err != nil {
		t.Fatalf("GetNode failed: %v", err)
	}
	if got.Found {
		t.Error("expected Found=false for missing node")
	}
}

func TestDeleteNode(t *testing.T) {
	client := startMockServer(t)
	if err := client.DeleteNode(context.Background(), "test-node-1", ""); err != nil {
		t.Fatalf("DeleteNode failed: %v", err)
	}
}

func TestUpdateEnergy(t *testing.T) {
	client := startMockServer(t)
	if err := client.UpdateEnergy(context.Background(), "test-node-1", 0.5, ""); err != nil {
		t.Fatalf("UpdateEnergy failed: %v", err)
	}
}

func TestInsertAndDeleteEdge(t *testing.T) {
	client := startMockServer(t)
	ctx := context.Background()

	id, err := client.InsertEdge(ctx, InsertEdgeOpts{
		From:     "node-a",
		To:       "node-b",
		EdgeType: "Association",
		Weight:   1.5,
	})
	if err != nil {
		t.Fatalf("InsertEdge failed: %v", err)
	}
	if id != "edge-auto-uuid" {
		t.Errorf("expected edge-auto-uuid, got %s", id)
	}

	if err := client.DeleteEdge(ctx, id, ""); err != nil {
		t.Fatalf("DeleteEdge failed: %v", err)
	}
}

func TestKnnSearch(t *testing.T) {
	client := startMockServer(t)
	results, err := client.KnnSearch(context.Background(), []float64{0.1, 0.2, 0.3}, 3, "")
	if err != nil {
		t.Fatalf("KnnSearch failed: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0].Distance != 0.0 {
		t.Errorf("expected first distance=0, got %f", results[0].Distance)
	}
}

func TestQuery(t *testing.T) {
	client := startMockServer(t)
	result, err := client.Query(context.Background(),
		"MATCH (n:Semantic) WHERE n.energy > $min RETURN n",
		map[string]interface{}{"min": 0.5},
		"",
	)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(result.Nodes))
	}
	if result.Explain == "" {
		t.Error("expected non-empty explain")
	}
}

func TestQueryParamTypes(t *testing.T) {
	client := startMockServer(t)
	_, err := client.Query(context.Background(),
		"MATCH (n) WHERE n.name = $name AND n.energy > $e RETURN n",
		map[string]interface{}{
			"name": "test",
			"e":    float64(0.5),
			"vec":  []float64{0.1, 0.2},
			"age":  int64(25),
		},
		"",
	)
	if err != nil {
		t.Fatalf("Query with mixed params failed: %v", err)
	}
}

func TestBfs(t *testing.T) {
	client := startMockServer(t)
	ids, err := client.Bfs(context.Background(), "node-a", TraversalOpts{MaxDepth: 5}, "")
	if err != nil {
		t.Fatalf("Bfs failed: %v", err)
	}
	if len(ids) != 3 {
		t.Errorf("expected 3 visited nodes, got %d", len(ids))
	}
}

func TestDijkstra(t *testing.T) {
	client := startMockServer(t)
	ids, costs, err := client.Dijkstra(context.Background(), "node-a", TraversalOpts{MaxDepth: 10, MaxCost: 5.0}, "")
	if err != nil {
		t.Fatalf("Dijkstra failed: %v", err)
	}
	if len(ids) != len(costs) {
		t.Errorf("ids and costs length mismatch: %d vs %d", len(ids), len(costs))
	}
}

func TestDiffuse(t *testing.T) {
	client := startMockServer(t)
	scales, err := client.Diffuse(context.Background(), []string{"node-a"}, DiffuseOpts{
		TValues:    []float64{0.1, 1.0},
		KChebyshev: 10,
	})
	if err != nil {
		t.Fatalf("Diffuse failed: %v", err)
	}
	if len(scales) != 2 {
		t.Errorf("expected 2 scales, got %d", len(scales))
	}
}

func TestCreateAndDropCollection(t *testing.T) {
	client := startMockServer(t)
	ctx := context.Background()

	created, err := client.CreateCollection(ctx, CollectionConfig{
		Name:   "test_col",
		Dim:    128,
		Metric: "poincare",
	})
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	if !created {
		t.Error("expected created=true")
	}

	if err := client.DropCollection(ctx, "test_col"); err != nil {
		t.Fatalf("DropCollection failed: %v", err)
	}
}

func TestListCollections(t *testing.T) {
	client := startMockServer(t)
	cols, err := client.ListCollections(context.Background())
	if err != nil {
		t.Fatalf("ListCollections failed: %v", err)
	}
	if len(cols) != 2 {
		t.Fatalf("expected 2 collections, got %d", len(cols))
	}
	if cols[1].Metric != "poincare" {
		t.Errorf("expected second collection metric=poincare, got %s", cols[1].Metric)
	}
}

func TestTriggerSleep(t *testing.T) {
	client := startMockServer(t)
	result, err := client.TriggerSleep(context.Background(), SleepOpts{
		Noise:     0.02,
		AdamSteps: 10,
	})
	if err != nil {
		t.Fatalf("TriggerSleep failed: %v", err)
	}
	if !result.Committed {
		t.Error("expected committed=true")
	}
	if result.HausdorffDelta >= 0 {
		t.Errorf("expected negative delta, got %f", result.HausdorffDelta)
	}
}

func TestInvokeZaratustra(t *testing.T) {
	client := startMockServer(t)
	result, err := client.InvokeZaratustra(context.Background(), ZaratustraOpts{
		Cycles: 1,
	})
	if err != nil {
		t.Fatalf("InvokeZaratustra failed: %v", err)
	}
	if result.NodesUpdated != 50 {
		t.Errorf("expected 50 nodes updated, got %d", result.NodesUpdated)
	}
	if len(result.EliteNodeIDs) != 2 {
		t.Errorf("expected 2 elite nodes, got %d", len(result.EliteNodeIDs))
	}
}

func TestInsertSensory(t *testing.T) {
	client := startMockServer(t)
	err := client.InsertSensory(context.Background(), InsertSensoryOpts{
		NodeID:         "sensory-node",
		Modality:       "audio",
		Latent:         []float32{0.1, 0.2, 0.3},
		OriginalBytes:  48000,
		EncoderVersion: 1,
	})
	if err != nil {
		t.Fatalf("InsertSensory failed: %v", err)
	}
}

func TestGetSensory(t *testing.T) {
	client := startMockServer(t)
	result, err := client.GetSensory(context.Background(), "sensory-node", "")
	if err != nil {
		t.Fatalf("GetSensory failed: %v", err)
	}
	if !result.Found {
		t.Error("expected found=true")
	}
	if result.Modality != "audio" {
		t.Errorf("expected modality=audio, got %s", result.Modality)
	}
}

func TestReconstruct(t *testing.T) {
	client := startMockServer(t)
	result, err := client.Reconstruct(context.Background(), "sensory-node", "full")
	if err != nil {
		t.Fatalf("Reconstruct failed: %v", err)
	}
	if !result.Found {
		t.Error("expected found=true")
	}
	if len(result.Latent) != 3 {
		t.Errorf("expected 3-d latent, got %d", len(result.Latent))
	}
}

func TestDegradeSensory(t *testing.T) {
	client := startMockServer(t)
	if err := client.DegradeSensory(context.Background(), "sensory-node", ""); err != nil {
		t.Fatalf("DegradeSensory failed: %v", err)
	}
}

func TestClose(t *testing.T) {
	client := startMockServer(t)
	if err := client.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func TestCloseNilConn(t *testing.T) {
	client := &NietzscheClient{}
	if err := client.Close(); err != nil {
		t.Fatalf("Close on nil conn should not error: %v", err)
	}
}

func TestDefaultCollectionEmpty(t *testing.T) {
	client := startMockServer(t)
	// Empty collection should work (server routes to "default")
	_, err := client.GetNode(context.Background(), "test-node-1", "")
	if err != nil {
		t.Fatalf("GetNode with empty collection failed: %v", err)
	}
}
