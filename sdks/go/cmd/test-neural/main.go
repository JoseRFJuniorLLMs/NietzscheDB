package main

import (
	"context"
	"fmt"
	"log"
	"time"

	nietzsche "nietzsche-sdk"
)

func main() {
	// Connect to local NietzscheDB
	client, err := nietzsche.ConnectInsecure("localhost:50051")
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("=== NietzscheDB Neural Integration Test ===")

	// 0. Ensure collection exists
	fmt.Print("Ensuring collection 'default' exists... ")
	_, err = client.CreateCollection(ctx, nietzsche.CollectionConfig{
		Name:   "default",
		Dim:    3072, // Use 3072 as expected by models
		Metric: "cosine",
	})
	if err != nil {
		fmt.Printf("FAILED (might already exist): %v\n", err)
	} else {
		fmt.Println("OK")
	}

	// 0b. Insert test nodes
	fmt.Print("Inserting test nodes... ")
	for _, id := range []string{"node-1", "node-2", "node-3"} {
		_, err = client.InsertNode(ctx, nietzsche.InsertNodeOpts{
			ID:         id,
			Collection: "default",
			NodeType:   "Semantic",
			Content:    map[string]string{"name": id},
		})
		if err != nil {
			fmt.Printf("FAILED to insert %s: %v\n", id, err)
		}
	}
	fmt.Println("OK")

	// 1. GNN Inference
	fmt.Print("Testing GnnInfer... ")
	nodeIDs := []string{"node-1", "node-2", "node-3"}
	gnnRes, err := client.GnnInfer(ctx, nietzsche.GnnInferOpts{
		ModelName:  "gnn_diffusion",
		Collection: "default",
		NodeIDs:    nodeIDs,
	})
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
	} else {
		fmt.Printf("OK (received %d embeddings)\n", len(gnnRes.Embeddings))
		if len(gnnRes.Embeddings) > 0 {
			fmt.Printf("Sample embedding: %v\n", gnnRes.Embeddings[0][:5])
		}
	}

	// 2. MCTS Search
	fmt.Print("Testing MctsSearch... ")
	mctsRes, err := client.MctsSearch(ctx, nietzsche.MctsOpts{
		ModelName:   "value_network",
		StartNodeID: "node-1",
		Simulations: 50,
		Collection:  "default",
	})
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
	} else {
		fmt.Printf("OK (best action: %s, value: %.4f)\n", mctsRes.BestActionID, mctsRes.Value)
	}

	// 3. Daemon CRUD
	fmt.Print("Testing Daemon CRUD... ")
	err = client.CreateDaemon(ctx, nietzsche.CreateDaemonOpts{
		Collection:   "default",
		Label:        "test-daemon",
		NQL:          "MATCH (n) RETURN n",
		IntervalSecs: 60,
	})
	if err != nil {
		fmt.Printf("Create FAILED: %v\n", err)
	} else {
		daemons, err := client.ListDaemons(ctx, "default")
		if err != nil {
			fmt.Printf("List FAILED: %v\n", err)
		} else {
			found := false
			for _, d := range daemons {
				if d.Label == "test-daemon" {
					found = true
					break
				}
			}
			if found {
				err = client.DropDaemon(ctx, "default", "test-daemon")
				if err != nil {
					fmt.Printf("Drop FAILED: %v\n", err)
				} else {
					fmt.Println("OK")
				}
			} else {
				fmt.Println("FAILED (daemon not listed)")
			}
		}
	}

	fmt.Println("=== Integration Test Complete ===")
}
