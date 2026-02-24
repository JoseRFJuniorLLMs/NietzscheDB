//! Reactive layer that converts agency events into executable intents.
//!
//! The reactor consumes `AgencyEvent`s from the event bus and produces
//! `AgencyIntent`s — declarative actions that the server executes under
//! a write lock. This keeps the agency crate itself read-only while
//! enabling automatic responses to graph health issues.
//!
//! ## Intent Flow
//!
//! ```text
//! Daemons → Events → Reactor → Intents → Server (write lock) → Graph mutations
//! ```

use tokio::sync::broadcast;
use uuid::Uuid;

use crate::config::AgencyConfig;
use crate::event_bus::{AgencyEvent, AgencyEventBus, WakeUpReason};
use crate::evolution::{EvolutionStrategy, RuleEvolution};
use crate::observer::HealthReport;

/// Declarative action produced by the reactor.
///
/// Intents are returned from the agency engine tick and executed by the
/// server under a write lock. The agency crate never mutates the graph
/// directly.
#[derive(Debug, Clone)]
pub enum AgencyIntent {
    /// Trigger a sleep cycle to reconsolidate embeddings.
    /// Produced when: EntropySpike detected or mean energy drops.
    TriggerSleepCycle {
        reason: String,
    },

    /// Trigger L-System growth to fill structural gaps.
    /// Produced when: CoherenceDrop detected or Hausdorff out of range.
    TriggerLSystemGrowth {
        reason: String,
    },

    /// Persist a HealthReport to RocksDB CF_META for historical tracking.
    PersistHealthReport {
        report: Box<HealthReport>,
    },

    /// Signal that a knowledge gap was detected in specific sectors.
    /// The server or EVA can use this to seek new knowledge.
    SignalKnowledgeGap {
        sectors: Vec<(usize, usize)>, // (depth_bin, angular_bin)
        suggested_depth_range: (f32, f32),
    },

    /// Trigger a directed dream from a seed node near a desire gap.
    /// Produced when: high-priority desire signal with available seed node.
    TriggerDream {
        seed_node_id: Uuid,
        depth: usize,
        reason: String,
    },

    /// Modulate Zaratustra alpha/decay based on health feedback.
    /// Produced when: HealthReport shows energy imbalance.
    ModulateZaratustra {
        alpha: f32,
        decay: f32,
        reason: String,
    },

    /// Generate a narrative report for the graph's current state.
    /// Produced when: HealthReport is generated (periodic).
    GenerateNarrative {
        collection: String,
        reason: String,
    },

    /// Evolve L-System rules based on health metrics.
    /// Produced when: HealthReport shows structural issues.
    EvolveLSystemRules {
        strategy: EvolutionStrategy,
        reason: String,
    },

    /// Trigger semantic GC: merge redundant nodes into an archetype.
    /// Produced when: NiilistaGcDaemon detects near-duplicate embeddings.
    TriggerSemanticGc {
        archetype_id: Uuid,
        redundant_ids: Vec<Uuid>,
    },

    /// Execute an autonomous NQL query (reflex).
    /// Produced when: ActionNode energy exceeds threshold and passes circuit breaker.
    ExecuteNQL {
        node_id: Uuid,
        nql: String,
        description: String,
    },

    /// Apply Long-Term Depression to an edge.
    /// Produced when: LTDDaemon detects correction_count > 0 on an edge.
    /// The server executes: `edge.weight = (edge.weight - weight_delta).max(0.0)`
    ApplyLtd {
        from_id: Uuid,
        to_id: Uuid,
        weight_delta: f32,
        correction_count: u64,
    },
    /// Boost a node's energy due to high neural structural importance.
    /// Produced when: NeuralProtection detected (GNN high-importance low-energy).
    NeuralBoost {
        node_id: Uuid,
        importance: f32,
        reason: String,
    },

    // ── Nezhmetdinov Forgetting Engine Intents ────────────────────

    /// Hard-delete a node from the graph (irreversible).
    /// Produced when: NezhmetdinovDaemon condemns a node via Triple Condition.
    HardDelete {
        node_id: Uuid,
        vitality: f32,
        reason: String,
    },

    /// Record a deletion in the Merkle-based audit ledger.
    /// Produced alongside each HardDelete for auditability.
    RecordDeletion {
        node_id: Uuid,
        cycle: u64,
        structural_hash: String,
    },
}

/// Converts agency events into executable intents.
///
/// The reactor subscribes to the event bus, drains events each tick,
/// and decides which intents to produce based on configured thresholds.
pub struct AgencyReactor {
    rx: broadcast::Receiver<AgencyEvent>,
    /// Cooldown: ticks since last sleep intent (avoid spamming).
    ticks_since_sleep_intent: u64,
    /// Cooldown: ticks since last lsystem intent.
    ticks_since_lsystem_intent: u64,
    /// PPO engine for neural policy-driven evolution.
    ppo_engine: Option<nietzsche_rl::PpoEngine>,
}

impl AgencyReactor {
    pub fn new(bus: &AgencyEventBus, ppo_engine: Option<nietzsche_rl::PpoEngine>) -> Self {
        Self {
            rx: bus.subscribe(),
            ticks_since_sleep_intent: u64::MAX / 2,    // allow immediate first trigger
            ticks_since_lsystem_intent: u64::MAX / 2,
            ppo_engine,
        }
    }

    /// Drain events and produce intents for this tick.
    ///
    /// Called once per agency engine tick, after all daemons and the
    /// observer have run.
    pub fn tick(&mut self, config: &AgencyConfig) -> Vec<AgencyIntent> {
        let mut intents = Vec::new();
        self.ticks_since_sleep_intent += 1;
        self.ticks_since_lsystem_intent += 1;

        // Drain all pending events
        let mut entropy_spikes = Vec::new();
        let mut coherence_drops = Vec::new();
        let mut knowledge_gaps = Vec::new();
        let mut health_reports = Vec::new();
        let mut wake_ups = Vec::new();
        let mut redundancies: Vec<(Uuid, Vec<Uuid>)> = Vec::new();
        let mut ltd_events: Vec<(Uuid, Uuid, f32, u64)> = Vec::new(); // (from, to, delta, count)
        let mut neural_protections: Vec<(Uuid, f32, String)> = Vec::new();

        loop {
            match self.rx.try_recv() {
                Ok(event) => match event {
                    AgencyEvent::EntropySpike { region_id, variance, .. } => {
                        entropy_spikes.push((region_id, variance));
                    }
                    AgencyEvent::CoherenceDrop { overlap_01_10, overlap_01_1 } => {
                        coherence_drops.push((overlap_01_10, overlap_01_1));
                    }
                    AgencyEvent::KnowledgeGap { sector, density, suggested_depth_range } => {
                        knowledge_gaps.push((sector, density, suggested_depth_range));
                    }
                    AgencyEvent::HealthReport(report) => {
                        health_reports.push(report);
                    }
                    AgencyEvent::SemanticRedundancy { archetype_id, redundant_ids, .. } => {
                        redundancies.push((archetype_id, redundant_ids));
                    }
                    AgencyEvent::TumorDetected { cluster_count, nodes_dampened, .. } => {
                        // Tumors detected and dampened — log for diagnostics.
                        // No additional intent needed; dampening was already applied.
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            "circuit breaker: {} tumors dampened ({} nodes)",
                            cluster_count,
                            nodes_dampened,
                        );
                        let _ = (cluster_count, nodes_dampened);
                    }
                    AgencyEvent::DaemonWakeUp { reason } => {
                        wake_ups.push(reason);
                    }
                    AgencyEvent::CorrectionAccumulated { from_id, to_id, weight_delta, correction_count } => {
                        ltd_events.push((from_id, to_id, weight_delta, correction_count));
                    }
                    AgencyEvent::NeuralProtection { node_id, importance, description } => {
                        neural_protections.push((node_id, importance, description));
                    }
                    // ── Nezhmetdinov Forgetting Engine events ─────────────
                    AgencyEvent::ForgettingCondemned { node_id, vitality, reason, .. } => {
                        intents.push(AgencyIntent::HardDelete {
                            node_id,
                            vitality,
                            reason,
                        });
                    }
                    AgencyEvent::ForgettingCycleComplete { .. } => {
                        // Logged for telemetry; no intent needed.
                    }
                },
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }

        let cooldown = config.reactor_cooldown_ticks;

        // ── Entropy spikes → trigger sleep cycle ──────────────────────────
        if !entropy_spikes.is_empty() && self.ticks_since_sleep_intent >= cooldown {
            let max_var = entropy_spikes.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
            intents.push(AgencyIntent::TriggerSleepCycle {
                reason: format!(
                    "entropy spike: {} regions, max_variance={:.3}",
                    entropy_spikes.len(),
                    max_var
                ),
            });
            self.ticks_since_sleep_intent = 0;
        }

        // ── Coherence drop → trigger L-System re-growth ──────────────────
        if !coherence_drops.is_empty() && self.ticks_since_lsystem_intent >= cooldown {
            let worst = coherence_drops
                .iter()
                .map(|(o, _)| *o)
                .fold(0.0f64, f64::max);
            intents.push(AgencyIntent::TriggerLSystemGrowth {
                reason: format!(
                    "coherence drop: overlap_01_10={:.3}",
                    worst
                ),
            });
            self.ticks_since_lsystem_intent = 0;
        }

        // ── Knowledge gaps → signal for external consumption ─────────────
        if !knowledge_gaps.is_empty() {
            let sectors: Vec<(usize, usize)> = knowledge_gaps
                .iter()
                .map(|(s, _, _)| (s.depth_bin, s.angular_bin))
                .collect();
            // Use the depth range of the first gap as a representative
            let depth_range = knowledge_gaps
                .first()
                .map(|(_, _, dr)| *dr)
                .unwrap_or((0.0, 1.0));
            intents.push(AgencyIntent::SignalKnowledgeGap {
                sectors,
                suggested_depth_range: depth_range,
            });
        }

        // ── Wake-up reasons → additional triggers ────────────────────────
        for reason in &wake_ups {
            match reason {
                WakeUpReason::MeanEnergyBelow(e) => {
                    if self.ticks_since_sleep_intent >= cooldown {
                        intents.push(AgencyIntent::TriggerSleepCycle {
                            reason: format!("mean energy below threshold: {:.3}", e),
                        });
                        self.ticks_since_sleep_intent = 0;
                    }
                }
                WakeUpReason::HausdorffOutOfRange(h) => {
                    if self.ticks_since_lsystem_intent >= cooldown {
                        intents.push(AgencyIntent::TriggerLSystemGrowth {
                            reason: format!("hausdorff out of range: {:.3}", h),
                        });
                        self.ticks_since_lsystem_intent = 0;
                    }
                }
                WakeUpReason::GapCountExceeded(count) => {
                    if self.ticks_since_lsystem_intent >= cooldown {
                        intents.push(AgencyIntent::TriggerLSystemGrowth {
                            reason: format!("excessive gaps: {} sectors empty", count),
                        });
                        self.ticks_since_lsystem_intent = 0;
                    }
                }
                WakeUpReason::TumorDetected { cluster_count, total_nodes } => {
                    // Tumor wake-up: trigger a sleep cycle to reconsolidate.
                    if self.ticks_since_sleep_intent >= cooldown {
                        intents.push(AgencyIntent::TriggerSleepCycle {
                            reason: format!(
                                "tumor detected: {} clusters, {} overheated nodes",
                                cluster_count, total_nodes
                            ),
                        });
                        self.ticks_since_sleep_intent = 0;
                    }
                }
            }
        }

        // ── Semantic redundancy → trigger GC merge ──────────────────────
        for (archetype_id, redundant_ids) in redundancies {
            intents.push(AgencyIntent::TriggerSemanticGc {
                archetype_id,
                redundant_ids,
            });
        }

        // ── LTD events → emit ApplyLTD intents ─────────────────────────────
        for (from_id, to_id, weight_delta, correction_count) in ltd_events {
            intents.push(AgencyIntent::ApplyLtd {
                from_id,
                to_id,
                weight_delta,
                correction_count,
            });
        }

        // ── Neural protections → emit NeuralBoost intents ──────────────────
        for (node_id, importance, reason) in neural_protections {
            intents.push(AgencyIntent::NeuralBoost {
                node_id,
                importance,
                reason,
            });
        }

        // ── Health reports → persist + modulate Zaratustra + evolve rules ─
        for report in health_reports {
            // Zaratustra modulation: adjust alpha/decay based on energy
            let (alpha, decay, reason) = compute_zaratustra_params(&report);
            intents.push(AgencyIntent::ModulateZaratustra {
                alpha,
                decay,
                reason,
            });

            // L-System evolution: suggest strategy based on health
            let mut strategy = RuleEvolution::suggest_strategy(&report);
            let mut reason_prefix = "heuristic";

            if let Some(ref ppo) = self.ppo_engine {
                // Feature vector: [mean_energy, is_fractal, gap_ratio, entropy_count/10, coherence]
                let gap_ratio = (report.gap_count as f32 / 80.0).min(1.0);
                let entropy_feat = (report.entropy_spike_count as f32 / 10.0).min(1.0);
                let features = vec![
                    report.mean_energy,
                    if report.is_fractal { 1.0 } else { 0.0 },
                    gap_ratio,
                    entropy_feat,
                    report.coherence_score as f32,
                ];
                let state = nietzsche_rl::GrowthState::new(features);
                
                let rt = tokio::runtime::Handle::current();
                match rt.block_on(ppo.suggest_action(&state)) {
                    Ok(action) => {
                        strategy = match action {
                            nietzsche_rl::GrowthAction::Balanced => crate::evolution::EvolutionStrategy::Balanced,
                            nietzsche_rl::GrowthAction::FavorGrowth => crate::evolution::EvolutionStrategy::FavorGrowth,
                            nietzsche_rl::GrowthAction::FavorPruning => crate::evolution::EvolutionStrategy::FavorPruning,
                            nietzsche_rl::GrowthAction::Consolidate => crate::evolution::EvolutionStrategy::Consolidate,
                        };
                        reason_prefix = "ppo";
                    }
                    Err(e) => tracing::warn!(error = %e, "PPO suggest_action failed, falling back to heuristic"),
                }
            }

            intents.push(AgencyIntent::EvolveLSystemRules {
                strategy,
                reason: format!(
                    "[{}] health: energy={:.2}, hausdorff={:.2}, gaps={}",
                    reason_prefix, report.mean_energy, report.global_hausdorff, report.gap_count
                ),
            });

            // Narrative generation
            intents.push(AgencyIntent::GenerateNarrative {
                collection: "default".to_string(),
                reason: format!("periodic health tick {}", report.tick_number),
            });

            intents.push(AgencyIntent::PersistHealthReport { report });
        }

        intents
    }
}

/// Compute Zaratustra alpha/decay from a health report.
///
/// - Low energy → increase alpha (inject more), decrease decay
/// - High energy → decrease alpha, increase decay (drain excess)
/// - Entropy spikes → increase decay (force reconsolidation)
/// - Healthy range → keep base values (α=0.10, δ=0.02)
fn compute_zaratustra_params(report: &HealthReport) -> (f32, f32, String) {
    let base_alpha = 0.10f32;
    let base_decay = 0.02f32;

    if report.mean_energy < 0.2 {
        // Critical low energy — boost alpha, minimize decay
        (
            (base_alpha * 2.0).min(0.25),
            (base_decay * 0.5).max(0.005),
            format!("energy critical ({:.2}) — boosting alpha", report.mean_energy),
        )
    } else if report.mean_energy < 0.4 {
        // Low energy — moderate boost
        (
            base_alpha * 1.5,
            base_decay * 0.75,
            format!("energy low ({:.2}) — increasing alpha", report.mean_energy),
        )
    } else if report.mean_energy > 0.85 {
        // Energy inflation — increase decay, decrease alpha
        (
            base_alpha * 0.5,
            (base_decay * 2.5).min(0.08),
            format!("energy inflated ({:.2}) — increasing decay", report.mean_energy),
        )
    } else if report.entropy_spike_count > 3 {
        // High entropy — boost decay for reconsolidation
        (
            base_alpha,
            (base_decay * 2.0).min(0.06),
            format!("entropy spikes ({}) — boosting decay", report.entropy_spike_count),
        )
    } else {
        // Healthy range — base parameters
        (base_alpha, base_decay, "healthy range — base params".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::SectorId;

    #[test]
    fn entropy_spike_triggers_sleep() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        bus.publish(AgencyEvent::EntropySpike {
            region_id: 0,
            variance: 0.5,
            sample_node_ids: vec![],
        });

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::TriggerSleepCycle { .. })));
    }

    #[test]
    fn coherence_drop_triggers_lsystem() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        bus.publish(AgencyEvent::CoherenceDrop {
            overlap_01_10: 0.85,
            overlap_01_1: 0.6,
        });

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::TriggerLSystemGrowth { .. })));
    }

    #[test]
    fn knowledge_gap_signals() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        bus.publish(AgencyEvent::KnowledgeGap {
            sector: SectorId { depth_bin: 2, angular_bin: 5 },
            density: 0.02,
            suggested_depth_range: (0.3, 0.5),
        });

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::SignalKnowledgeGap { .. })));
    }

    #[test]
    fn cooldown_prevents_spam() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig {
            reactor_cooldown_ticks: 3,
            ..AgencyConfig::default()
        };
        let mut reactor = AgencyReactor::new(&bus, None);

        // First spike → intent
        bus.publish(AgencyEvent::EntropySpike {
            region_id: 0,
            variance: 0.5,
            sample_node_ids: vec![],
        });
        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::TriggerSleepCycle { .. })));

        // Second spike immediately → no intent (cooldown)
        bus.publish(AgencyEvent::EntropySpike {
            region_id: 1,
            variance: 0.6,
            sample_node_ids: vec![],
        });
        let intents = reactor.tick(&config);
        assert!(!intents.iter().any(|i| matches!(i, AgencyIntent::TriggerSleepCycle { .. })));

        // After cooldown → intent again
        for _ in 0..3 {
            let _ = reactor.tick(&config);
        }
        bus.publish(AgencyEvent::EntropySpike {
            region_id: 2,
            variance: 0.7,
            sample_node_ids: vec![],
        });
        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::TriggerSleepCycle { .. })));
    }

    #[test]
    fn health_report_always_persisted() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        let report = HealthReport::default();
        bus.publish(AgencyEvent::HealthReport(Box::new(report)));

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::PersistHealthReport { .. })));
    }

    #[test]
    fn wake_up_low_energy_triggers_sleep() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        bus.publish(AgencyEvent::DaemonWakeUp {
            reason: WakeUpReason::MeanEnergyBelow(0.15),
        });

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::TriggerSleepCycle { .. })));
    }

    #[test]
    fn health_report_triggers_zaratustra_modulation() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        let report = HealthReport {
            mean_energy: 0.15, // Very low
            ..HealthReport::default()
        };
        bus.publish(AgencyEvent::HealthReport(Box::new(report)));

        let intents = reactor.tick(&config);
        let modulate = intents.iter().find(|i| matches!(i, AgencyIntent::ModulateZaratustra { .. }));
        assert!(modulate.is_some(), "should produce ModulateZaratustra");
        if let Some(AgencyIntent::ModulateZaratustra { alpha, decay, .. }) = modulate {
            assert!(*alpha > 0.10, "low energy should boost alpha");
            assert!(*decay < 0.02, "low energy should reduce decay");
        }
    }

    #[test]
    fn health_report_triggers_evolution() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        let report = HealthReport {
            mean_energy: 0.6,
            is_fractal: true,
            gap_count: 30, // High gaps
            ..HealthReport::default()
        };
        bus.publish(AgencyEvent::HealthReport(Box::new(report)));

        let intents = reactor.tick(&config);
        let evolve = intents.iter().find(|i| matches!(i, AgencyIntent::EvolveLSystemRules { .. }));
        assert!(evolve.is_some(), "should produce EvolveLSystemRules");
        if let Some(AgencyIntent::EvolveLSystemRules { strategy, .. }) = evolve {
            assert_eq!(*strategy, EvolutionStrategy::FavorGrowth);
        }
    }

    #[test]
    fn health_report_triggers_narrative() {
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut reactor = AgencyReactor::new(&bus, None);

        bus.publish(AgencyEvent::HealthReport(Box::new(HealthReport::default())));

        let intents = reactor.tick(&config);
        assert!(intents.iter().any(|i| matches!(i, AgencyIntent::GenerateNarrative { .. })));
    }

    #[test]
    fn zaratustra_params_healthy() {
        let report = HealthReport {
            mean_energy: 0.6,
            entropy_spike_count: 0,
            ..HealthReport::default()
        };
        let (alpha, decay, _) = compute_zaratustra_params(&report);
        assert!((alpha - 0.10).abs() < 0.01, "healthy → base alpha");
        assert!((decay - 0.02).abs() < 0.01, "healthy → base decay");
    }

    #[test]
    fn zaratustra_params_inflated() {
        let report = HealthReport {
            mean_energy: 0.95,
            ..HealthReport::default()
        };
        let (alpha, decay, _) = compute_zaratustra_params(&report);
        assert!(alpha < 0.10, "inflated energy → lower alpha");
        assert!(decay > 0.02, "inflated energy → higher decay");
    }
}
