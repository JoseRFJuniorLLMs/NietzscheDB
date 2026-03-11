# NietzscheDB Neural Networks

NietzscheDB embeds **12 small neural networks** that run at inference time via ONNX Runtime.
All models are trained in Python (PyTorch) and exported to ONNX for zero-dependency Rust inference.

## Network Inventory

| # | Network | Crate | Input | Output | Purpose |
|---|---------|-------|-------|--------|---------|
| 1 | **GNN Diffusion** | `nietzsche-gnn` | [B, 3072] node features | [B, 3072] + [B, 1] | Refine embeddings + predict node importance |
| 2 | **VQ-VAE** | `nietzsche-vqvae` | [B, 3072] embeddings | [B, 3072] recon + VQ loss | Latent compression for DSI indexing |
| 3 | **PPO Actor** | `nietzsche-rl` | [B, 64] graph state | [B, 4] action probs | L-System growth strategy selection |
| 4 | **Value Network** | `nietzsche-mcts` | [B, 64] graph state | [B, 1] value score | MCTS leaf evaluation |
| 5 | **Edge Predictor** | `nietzsche-gnn` | [B, 256] node pair | [B, 1] edge prob | Link prediction for topology optimization |
| 6 | **Image Encoder** | `nietzsche-sensory` | [B, 3, 64, 64] RGB | [B, 128] latent | CNN image → Poincaré ball |
| 7 | **Audio Encoder** | `nietzsche-sensory` | [B, 1, 64, 32] mel | [B, 128] latent | Conv audio → Poincaré ball |
| 8 | **Dream Generator** | `nietzsche-dream` | [B, 192] seed+noise | [B, 128] generated | Synthesize new node embeddings |
| 9 | **Cluster Scorer** | `nietzsche-cluster` | [B, 261] cluster stats | [B, 3] keep/split/merge | Cluster health evaluation |
| 10 | **DSI Decoder** | `nietzsche-dsi` | [B, 128] query | [B, 4, 1024] logits | Neural document retrieval |
| 11 | **Anomaly Detector** | `nietzsche-wiederkehr` | [B, 64] health state | [B, 65] recon+score | Detect degenerative patterns |
| 12 | **Structural Evolver** | `nietzsche-neural` | varies | varies | Structural evolution policy |

## Architecture Principles

1. **Inference-only in Rust**: All networks run via ONNX Runtime (`ort` crate). No training in production.
2. **GPU-accelerated**: All 12 models use `CUDAExecutionProvider` for inference on the NVIDIA L4 GPU. The `ort/cuda` feature is enabled via `nietzsche-neural`'s `cuda` feature, wired through the server's `gpu` feature flag. Without the `cuda` feature, `ort` silently falls back to CPU.
3. **Hyperbolic-aware**: Outputs are projected to the Poincaré ball via `exp_map_zero` where needed.
4. **Small by design**: Each network has < 1M parameters. Total neural footprint < 10MB ONNX.
5. **Self-supervised**: Most networks can train without labeled data (contrastive, reconstruction, self-play).
6. **Thread-safe**: Models are managed by `ModelRegistry` with `Arc<Mutex<Session>>`.

## GPU Inference Configuration

All models are loaded through `nietzsche_neural::REGISTRY.load_model()` which creates an ONNX session with `CUDAExecutionProvider`:

```rust
// nietzsche-neural/src/lib.rs
let session = Session::builder()?
    .with_execution_providers([CUDAExecutionProvider::default().build()])?
    .commit_from_file(&meta.path)?;
```

**Feature flags** (Cargo.toml):
- `nietzsche-neural`: `cuda = ["ort/cuda"]` — enables CUDA execution provider in `ort`
- `nietzsche-server`: `gpu = [..., "nietzsche-neural/cuda"]` — propagates CUDA to all models

**Runtime requirements**: CUDA 12.x + ONNX Runtime with CUDA support. Models are loaded from `NIETZSCHE_MODEL_DIR` (default: `/var/lib/nietzsche/models/`).

**Startup log** confirms GPU inference per model:
```
INFO nietzsche_neural: Model loaded with CUDA execution provider (GPU) model=vqvae
INFO nietzsche_neural: Model loaded with CUDA execution provider (GPU) model=dsi_decoder
...
```

## Training Pipeline

All training scripts are in `scripts/models/`:

```
scripts/models/
├── train_gnn.py              # GNN Diffusion (MSE + BCE)
├── train_vqvae.py             # VQ-VAE (reconstruction + codebook)
├── train_ppo.py               # PPO Actor-Critic (self-play)
├── train_value_network.py     # MCTS Value Network (MSE)
├── train_edge_predictor.py    # Edge Predictor (binary classification)
├── train_image_encoder.py     # Image Encoder (NT-Xent contrastive)
├── train_audio_encoder.py     # Audio Encoder (reconstruction)
├── train_dream_generator.py   # Dream Generator (MMD + Poincaré)
├── train_cluster_scorer.py    # Cluster Scorer (cross-entropy)
├── train_dsi_decoder.py       # DSI Decoder (cross-entropy per level)
├── train_anomaly_detector.py  # Anomaly Detector (2-phase: recon + classify)
├── export_gnn.py              # GNN architecture definition
├── export_value_network.py    # Value Network architecture
├── export_image_encoder.py    # Image Encoder architecture
├── export_audio_encoder.py    # Audio Encoder architecture
├── export_dream_generator.py  # Dream Generator architecture
├── export_cluster_scorer.py   # Cluster Scorer architecture
├── export_edge_predictor.py   # Edge Predictor architecture
├── export_dsi_decoder.py      # DSI Decoder architecture
└── export_anomaly_detector.py # Anomaly Detector architecture
```

### How to Train

```bash
# Prerequisites
pip install torch numpy onnx

# Train all networks (uses synthetic data if no real data available)
cd scripts/models
python train_gnn.py
python train_vqvae.py
python train_ppo.py
python train_value_network.py
python train_edge_predictor.py
python train_image_encoder.py
python train_audio_encoder.py
python train_dream_generator.py
python train_cluster_scorer.py
python train_dsi_decoder.py
python train_anomaly_detector.py
```

Checkpoints are saved to `checkpoints/` (PyTorch `.pt`).
ONNX models are exported to `models/` (ready for Rust inference).

### Training with Real Data

Each training script accepts a `data_path` pointing to a `.pt` file.
Use the Go distiller or export scripts to generate real training data:

| Network | Data File | Content |
|---------|-----------|---------|
| GNN | `clinical_dataset.pt` | Node embeddings (3072D) + importance labels |
| VQ-VAE | `clinical_dataset.pt` | Node embeddings (3072D) |
| PPO | `health_trajectories.pt` | Graph health state sequences |
| Value Net | `clinical_dataset.pt` | States projected to 64D |
| Edge Predictor | `edge_dataset.pt` | Node pair embeddings + edge labels |
| Image Encoder | `image_dataset.pt` | Image patches [N, 3, 64, 64] |
| Audio Encoder | `audio_dataset.pt` | Mel spectrograms [N, 1, 64, 32] |
| Dream Generator | `clinical_dataset.pt` | Node embeddings (projected to 128D) |
| Cluster Scorer | `cluster_dataset.pt` | Cluster statistics + labels |
| DSI Decoder | `dsi_dataset.pt` | Query embeddings + VQ code targets |
| Anomaly Detector | `health_trajectories.pt` | Normal health state distributions |

## Rust Inference API

```rust
use nietzsche_neural::REGISTRY;

// Edge Predictor
let predictor = nietzsche_gnn::EdgePredictorNet::new("./models");
let prob = predictor.predict(&embedding_a, &embedding_b)?;

// Dream Generator
let dreamer = nietzsche_dream::DreamGeneratorNet::new("./models");
let new_embedding = dreamer.dream(&seed_embedding, 0.5)?; // creativity = 0.5

// Anomaly Detector
let detector = nietzsche_wiederkehr::AnomalyDetectorNet::new("./models")
    .with_threshold(0.6);
let result = detector.detect(&health_state)?;
if result.is_anomalous {
    tracing::warn!("anomaly detected: score={}", result.combined_score);
}

// Cluster Scorer
let scorer = nietzsche_cluster::ClusterScorerNet::new("./models");
let score = scorer.score(&centroid, &variance, &[size, density, weight, diam, coh])?;
match score.action {
    ClusterAction::Split => { /* split cluster */ }
    ClusterAction::Merge => { /* merge with neighbor */ }
    ClusterAction::Keep  => { /* no action */ }
}

// DSI Decoder (neural retrieval)
let dsi = nietzsche_dsi::DsiDecoderNet::new("./models");
let result = dsi.decode(&query_embedding)?;
// result.codes = [42, 817, 3, 501] → hierarchical node address

// MCTS Value Network
let value_net = nietzsche_mcts::ValueNetworkInference::new("./models");
let score = value_net.evaluate(&graph_state)?; // 0.0 = bad, 1.0 = excellent
```
