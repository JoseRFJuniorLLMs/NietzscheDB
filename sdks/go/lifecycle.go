// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// TriggerSleep initiates a Riemannian reconsolidation sleep cycle.
// During sleep, the database perturbs node embeddings via Adam optimisation
// and commits only if the Hausdorff distance delta stays within threshold.
func (c *NietzscheClient) TriggerSleep(ctx context.Context, opts SleepOpts) (SleepResult, error) {
	resp, err := c.stub.TriggerSleep(ctx, &pb.SleepRequest{
		Noise:              opts.Noise,
		AdamSteps:          opts.AdamSteps,
		AdamLr:             opts.AdamLr,
		HausdorffThreshold: opts.HausdorffThreshold,
		RngSeed:            opts.RngSeed,
		Collection:         opts.Collection,
	})
	if err != nil {
		return SleepResult{}, fmt.Errorf("nietzsche TriggerSleep: %w", err)
	}

	return SleepResult{
		HausdorffBefore: resp.HausdorffBefore,
		HausdorffAfter:  resp.HausdorffAfter,
		HausdorffDelta:  resp.HausdorffDelta,
		Committed:       resp.Committed,
		NodesPerturbed:  resp.NodesPerturbed,
		SnapshotNodes:   resp.SnapshotNodes,
	}, nil
}

// InvokeZaratustra runs the three-phase autonomous evolution cycle:
//   - Phase 1: Will to Power — energy propagation
//   - Phase 2: Eternal Recurrence — temporal echo snapshots
//   - Phase 3: Übermensch — elite tier identification
func (c *NietzscheClient) InvokeZaratustra(ctx context.Context, opts ZaratustraOpts) (*ZaratustraResult, error) {
	resp, err := c.stub.InvokeZaratustra(ctx, &pb.ZaratustraRequest{
		Collection: opts.Collection,
		Alpha:      opts.Alpha,
		Decay:      opts.Decay,
		Cycles:     opts.Cycles,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche InvokeZaratustra: %w", err)
	}

	return &ZaratustraResult{
		NodesUpdated:     resp.NodesUpdated,
		MeanEnergyBefore: resp.MeanEnergyBefore,
		MeanEnergyAfter:  resp.MeanEnergyAfter,
		TotalEnergyDelta: resp.TotalEnergyDelta,
		EchoesCreated:    resp.EchoesCreated,
		EchoesEvicted:    resp.EchoesEvicted,
		TotalEchoes:      resp.TotalEchoes,
		EliteCount:       resp.EliteCount,
		EliteThreshold:   resp.EliteThreshold,
		MeanEliteEnergy:  resp.MeanEliteEnergy,
		MeanBaseEnergy:   resp.MeanBaseEnergy,
		EliteNodeIDs:     resp.EliteNodeIds,
		DurationMs:       resp.DurationMs,
		CyclesRun:        resp.CyclesRun,
	}, nil
}
