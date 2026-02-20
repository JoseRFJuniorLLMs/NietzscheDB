# sdk-papa-caolho

Go SDK for NietzscheDB — Temporal Hyperbolic Graph Database.

## Install

```bash
go get sdk-papa-caolho@latest
```

Or use a local replace directive during development:

```go
// go.mod
replace sdk-papa-caolho => ../Nietzsche-Database/sdks/go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    nietzsche "sdk-papa-caolho"
)

func main() {
    client, err := nietzsche.ConnectInsecure("localhost:50052")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Health check
    if err := client.HealthCheck(ctx); err != nil {
        log.Fatal(err)
    }

    // Create a collection
    client.CreateCollection(ctx, nietzsche.CollectionConfig{
        Name:   "memories",
        Dim:    128,
        Metric: "poincare",
    })

    // Insert a node
    node, _ := client.InsertNode(ctx, nietzsche.InsertNodeOpts{
        Coords:     make([]float64, 128),
        Content:    map[string]string{"text": "first memory"},
        NodeType:   "Semantic",
        Collection: "memories",
    })
    fmt.Println("Inserted node:", node.ID)

    // KNN search
    results, _ := client.KnnSearch(ctx, make([]float64, 128), 5, "memories")
    for _, r := range results {
        fmt.Printf("  %s (dist=%.4f)\n", r.ID, r.Distance)
    }

    // NQL query
    qr, _ := client.Query(ctx,
        "MATCH (n:Semantic) WHERE n.energy > $min RETURN n",
        map[string]interface{}{"min": 0.5},
        "memories",
    )
    fmt.Printf("Query returned %d nodes\n", len(qr.Nodes))

    // Trigger sleep cycle
    sleep, _ := client.TriggerSleep(ctx, nietzsche.SleepOpts{Collection: "memories"})
    fmt.Printf("Sleep: H %.4f -> %.4f (committed=%v)\n",
        sleep.HausdorffBefore, sleep.HausdorffAfter, sleep.Committed)
}
```

## API Coverage

All 22 gRPC RPCs are covered:

| Group | Methods |
|-------|---------|
| Collections | `CreateCollection`, `DropCollection`, `ListCollections` |
| Nodes | `InsertNode`, `GetNode`, `DeleteNode`, `UpdateEnergy` |
| Edges | `InsertEdge`, `DeleteEdge` |
| Query | `Query` (NQL), `KnnSearch` |
| Traversal | `Bfs`, `Dijkstra`, `Diffuse` |
| Lifecycle | `TriggerSleep`, `InvokeZaratustra` |
| Sensory | `InsertSensory`, `GetSensory`, `Reconstruct`, `DegradeSensory` |
| Admin | `GetStats`, `HealthCheck` |

## Proto Regeneration

```bash
make proto
```

Requires `protoc`, `protoc-gen-go`, and `protoc-gen-go-grpc`.

## Tests

```bash
make test
```

Tests use `bufconn` (in-memory gRPC) — no running NietzscheDB instance required.

## License

AGPL-3.0-or-later
