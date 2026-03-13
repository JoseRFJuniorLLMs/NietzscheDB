# GeometricKernels × NietzscheDB — Roadmap de Integração

> **Data**: 2026-03-12
> **Biblioteca**: [GeometricKernels v1.0](https://github.com/geometric-kernels/GeometricKernels)
> **Localização local**: `d:/DEV/GeometricKernels/`

---

## Visão Geral

O NietzscheDB já implementa difusão de calor via polinómios de Chebyshev (`nietzsche-pregel`),
operações hiperbólicas multi-manifold (`nietzsche-hyp-ops`), e métricas epistémicas
(`nietzsche-epistemics`). A GeometricKernels fornece implementações **matematicamente exatas**
de kernels de Matérn e Heat em variedades (grafos, espaço hiperbólico, esferas, grupos de Lie).

A integração **não substitui** nada existente — adiciona um **motor matemático auxiliar** em Python
que calibra, valida e estende as aproximações Rust.

---

## Arquitectura de Integração

```
┌─────────────────────────────────────────────────────────┐
│                    NietzscheDB (Rust)                    │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ nietzsche-   │  │ nietzsche-   │  │ nietzsche-    │  │
│  │ pregel       │  │ hyp-ops      │  │ query (NQL)   │  │
│  │ (Chebyshev)  │  │ (Poincaré)   │  │ (MathFunc)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
│         │                 │                 │            │
│  ┌──────┴─────────────────┴─────────────────┴────────┐  │
│  │            nietzsche-neural (ONNX Runtime)         │  │
│  │            ModelRegistry + infer_f32()             │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │ .onnx models                   │
└─────────────────────────┼───────────────────────────────┘
                          │
              ┌───────────┴───────────────┐
              │  GeometricKernels (Python) │
              │                           │
              │  ┌─────────────────────┐  │
              │  │ Spaces:             │  │
              │  │  - Graph            │  │
              │  │  - Hyperbolic       │  │
              │  │  - Hypersphere      │  │
              │  │  - ProductSpace     │  │
              │  └─────────┬───────────┘  │
              │            │              │
              │  ┌─────────┴───────────┐  │
              │  │ Kernels:            │  │
              │  │  - MaternKL         │  │
              │  │  - MaternFeatureMap │  │
              │  │  - HeatKernel(ν=∞)  │  │
              │  └─────────┬───────────┘  │
              │            │              │
              │  ┌─────────┴───────────┐  │
              │  │ GP Frontends:       │  │
              │  │  - GPyTorch         │  │
              │  │  - GPJax            │  │
              │  └─────────────────────┘  │
              └───────────────────────────┘
```

**Ponte principal**: Python treina/calibra → exporta ONNX → Rust executa em runtime.

---

## Fases de Implementação

### Fase 1 — Fundação Python (Semana 1-2)

**Objetivo**: Serviço Python que expõe GeometricKernels sobre grafos NietzscheDB via gRPC.

#### 1.1 Criar `nietzsche-lab/geometric_service/`

```
nietzsche-lab/geometric_service/
├── __init__.py
├── graph_bridge.py        # NietzscheDB graph → GeometricKernels Graph space
├── kernel_service.py      # Serviço principal (Heat, Matérn, GP)
├── calibration.py         # Comparação Chebyshev vs GeometricKernels exato
├── uncertainty.py          # GP-based uncertainty estimation
├── export_onnx.py         # Export feature maps para ONNX
└── requirements.txt       # geometric-kernels, gpytorch, onnx, grpcio
```

#### 1.2 Graph Bridge (`graph_bridge.py`)

Converter o grafo NietzscheDB para o formato GeometricKernels:

```python
import numpy as np
from scipy.sparse import csr_matrix
from geometric_kernels.spaces import Graph
from nietzsche_lab.grpc_client import NietzscheClient

def ndb_to_geometric_graph(client: NietzscheClient, collection: str) -> Graph:
    """
    Extrai grafo do NietzscheDB e cria espaço GeometricKernels.

    O GeometricKernels Graph aceita uma adjacency matrix (scipy sparse).
    Usa normalize_laplacian=True para consistência com nietzsche-pregel.
    """
    nodes, edges = client.sample_subgraph(limit=5000)

    # Construir mapa de índices
    node_ids = [n.id for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    # Construir adjacency matrix esparsa
    rows, cols, weights = [], [], []
    for e in edges:
        if e.from_id in id_to_idx and e.to_id in id_to_idx:
            i, j = id_to_idx[e.from_id], id_to_idx[e.to_id]
            w = e.weight if e.weight > 0 else 1.0
            rows.extend([i, j])  # undirected
            cols.extend([j, i])
            weights.extend([w, w])

    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # num_eigenpairs controla a truncagem espectral
    # Mais eigenpairs = mais preciso, mas mais lento
    space = Graph(adj, normalize_laplacian=True)

    return space, node_ids, id_to_idx
```

#### 1.3 Kernel Service (`kernel_service.py`)

```python
import numpy as np
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Graph, Hyperbolic

class NietzscheKernelService:
    """
    Serviço de kernels geométricos para NietzscheDB.

    Três modos:
    1. heat_diffusion  — kernel de calor exato no grafo (ν=∞)
    2. matern          — kernel Matérn com smoothness controlável
    3. uncertainty     — GP posterior com variância (incerteza cognitiva)
    """

    def __init__(self, graph_space: Graph):
        self.space = graph_space
        self.kernel = MaternGeometricKernel(graph_space)
        self.params = self.kernel.init_params()

    def heat_kernel_matrix(self, node_indices: np.ndarray, t: float) -> np.ndarray:
        """
        Calcula matriz de kernel de calor K_t(i,j) para nós selecionados.

        Equivalente ao e^{-tL} do nietzsche-pregel, mas exato (não aproximado).
        ν=∞ → heat kernel; lengthscale = √(2t)
        """
        self.params["nu"] = np.array([np.inf])
        self.params["lengthscale"] = np.array([np.sqrt(2 * t)])

        X = node_indices.reshape(-1, 1).astype(np.int64)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def matern_kernel_matrix(self, node_indices: np.ndarray,
                              nu: float = 2.5, lengthscale: float = 1.0) -> np.ndarray:
        """
        Kernel Matérn — mais flexível que heat kernel.

        nu controla suavidade:
          - nu=0.5  → very rough (Laplacian kernel)
          - nu=1.5  → once differentiable
          - nu=2.5  → twice differentiable (bom default)
          - nu=∞    → heat kernel (infinitely smooth)
        """
        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([lengthscale])

        X = node_indices.reshape(-1, 1).astype(np.int64)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def activation_vector(self, source_indices: list[int], t: float) -> np.ndarray:
        """
        Dado nós-fonte, calcula ativação difundida para todos os nós.

        Equivalente direto do DiffusionEngine::diffuse() em Rust,
        mas usando kernel exato em vez de Chebyshev.
        """
        all_indices = np.arange(self.space.num_vertices).reshape(-1, 1)
        sources = np.array(source_indices).reshape(-1, 1)

        self.params["nu"] = np.array([np.inf])
        self.params["lengthscale"] = np.array([np.sqrt(2 * t)])

        # K[i, j] = heat_kernel(all_node_i, source_j)
        K = self.kernel.K(self.params, all_indices, sources)

        # Soma ativação de todas as fontes
        activation = np.array(K).sum(axis=1)
        return activation
```

#### 1.4 Ficheiros de Entrega — Fase 1

| Ficheiro | Descrição |
|----------|-----------|
| `graph_bridge.py` | NietzscheDB → GeometricKernels Graph |
| `kernel_service.py` | Heat/Matérn/Ativação sobre o grafo |
| `requirements.txt` | `geometric-kernels>=1.0`, `gpytorch`, `scipy` |
| `test_gk_basic.py` | Teste: criar espaço, computar kernel, verificar simetria |

---

### Fase 2 — Calibração do Chebyshev (Semana 3-4)

**Objetivo**: Usar kernels exatos da GeometricKernels como ground truth para calibrar
e validar a aproximação Chebyshev do `nietzsche-pregel`.

#### 2.1 Calibração (`calibration.py`)

```python
import numpy as np
from .kernel_service import NietzscheKernelService
from .graph_bridge import ndb_to_geometric_graph

class ChebyshevCalibrator:
    """
    Compara difusão Chebyshev (Rust) vs Heat Kernel exato (GeometricKernels).

    Objectivo: encontrar K_max ótimo e validar erro de aproximação.
    """

    def __init__(self, gk_service: NietzscheKernelService, ndb_client):
        self.gk = gk_service
        self.ndb = ndb_client

    def compare_diffusion(self, source_node_id: str, t_values: list[float],
                          k_max_values: list[int] = [10, 20, 30, 50]) -> dict:
        """
        Para cada (t, k_max), compara:
        - activation_chebyshev (via NietzscheDB gRPC diffuse)
        - activation_exact (via GeometricKernels)

        Métricas:
        - L2 error normalizado
        - Rank correlation (Spearman) — os top-K activados são os mesmos?
        - Overlap@K — quantos dos top-K coincidem?
        """
        results = {}
        for t in t_values:
            exact = self.gk.activation_vector([source_idx], t)
            for k_max in k_max_values:
                chebyshev = self.ndb.diffuse_chebyshev(source_node_id, t, k_max)

                # Normalizar para comparação
                exact_norm = exact / (exact.max() + 1e-15)
                cheb_norm = chebyshev / (chebyshev.max() + 1e-15)

                l2_error = np.linalg.norm(exact_norm - cheb_norm) / len(exact)

                # Rank correlation
                from scipy.stats import spearmanr
                rho, _ = spearmanr(exact_norm, cheb_norm)

                # Overlap@20
                top_exact = set(np.argsort(-exact_norm)[:20])
                top_cheb = set(np.argsort(-cheb_norm)[:20])
                overlap = len(top_exact & top_cheb) / 20

                results[(t, k_max)] = {
                    "l2_error": l2_error,
                    "spearman_rho": rho,
                    "overlap_at_20": overlap,
                }
        return results

    def find_optimal_k_max(self, target_overlap: float = 0.9) -> dict[float, int]:
        """
        Para cada escala t, encontra o K_max mínimo que atinge o overlap target.

        Resultado esperado:
        - t=0.1 (focused): K_max ≈ 10-15 basta
        - t=1.0 (associative): K_max ≈ 20-30
        - t=10.0 (free association): K_max ≈ 40-50+
        """
        pass  # implementar com binary search sobre k_max_values
```

#### 2.2 Relatório de Calibração

O output desta fase é um relatório que responde:

1. **Qual o erro real da aproximação Chebyshev?** (L2, Spearman, Overlap)
2. **K_max ótimo para cada escala cognitiva** (t=0.1, 1.0, 10.0)
3. **Onde a aproximação falha** — que topologias de grafo causam mais erro?
4. **A ponderação por valência afecta a precisão?** (comparar com/sem `valence_modifier`)

---

### Fase 3 — Motor de Incerteza Cognitiva (Semana 5-7)

**Objetivo**: Gaussian Processes sobre o grafo para detectar lacunas de conhecimento.

#### 3.1 GP-based Uncertainty (`uncertainty.py`)

```python
import numpy as np
import torch
import gpytorch
from geometric_kernels.frontends import GPyTorchGeometricKernel
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Graph

class CognitiveUncertaintyModel(gpytorch.models.ExactGP):
    """
    Gaussian Process sobre o grafo de conhecimento.

    Input: índice do nó (posição no grafo)
    Output: "confiança epistémica" — quão bem o sistema conhece esta região

    Alta variância posterior = lacuna de conhecimento = target para EpistemologyDaemon
    """

    def __init__(self, train_x, train_y, likelihood, graph_space: Graph):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = MaternGeometricKernel(graph_space)
        self.covar_module = GPyTorchGeometricKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EpistemicUncertaintyEstimator:
    """
    Estima incerteza epistémica sobre o grafo NietzscheDB.

    Workflow:
    1. Extrai nós com energia > threshold como "observações" (y = energy)
    2. Treina GP com kernel geométrico sobre o grafo
    3. Prediz variância posterior para TODOS os nós
    4. Alta variância = lacuna epistémica
    """

    def __init__(self, graph_space: Graph, node_ids: list[str]):
        self.space = graph_space
        self.node_ids = node_ids
        self.model = None
        self.likelihood = None

    def fit(self, observed_indices: np.ndarray, observed_values: np.ndarray,
            training_iterations: int = 50):
        """
        Treina GP usando nós observados (com energia/ativação conhecida).
        """
        train_x = torch.tensor(observed_indices, dtype=torch.long).unsqueeze(-1)
        train_y = torch.tensor(observed_values, dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = CognitiveUncertaintyModel(
            train_x, train_y, self.likelihood, self.space
        )

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict_uncertainty(self) -> dict[str, float]:
        """
        Retorna variância posterior para cada nó.

        Resultado: {node_id: variance}
        Nós com alta variância são candidatos para:
        - EpistemologyDaemon (pesquisa autónoma)
        - Evolution27 (mutação epistémica)
        - AgencyIntent::EpistemicMutation
        """
        self.model.eval()
        self.likelihood.eval()

        all_indices = torch.arange(len(self.node_ids)).unsqueeze(-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(all_indices))
            means = pred.mean.numpy()
            variances = pred.variance.numpy()

        return {
            self.node_ids[i]: {
                "mean": float(means[i]),
                "variance": float(variances[i]),
                "uncertainty": float(np.sqrt(variances[i])),
            }
            for i in range(len(self.node_ids))
        }

    def find_knowledge_gaps(self, top_k: int = 20) -> list[dict]:
        """
        Retorna os top-K nós com maior incerteza epistémica.

        Estes são candidatos para:
        - Pesquisa autónoma (NietzscheLab)
        - Criação de novas arestas (Evolution27)
        - Energia boost (Agency reactor)
        """
        uncertainties = self.predict_uncertainty()
        sorted_nodes = sorted(
            uncertainties.items(),
            key=lambda x: x[1]["variance"],
            reverse=True
        )
        return [
            {"node_id": nid, **data}
            for nid, data in sorted_nodes[:top_k]
        ]
```

#### 3.2 Integração com NietzscheLab

Modificar `nietzsche-lab/lab_runner.py` para usar incerteza como guia:

```python
# Em lab_runner.py, após sample_subgraph():

from geometric_service.uncertainty import EpistemicUncertaintyEstimator
from geometric_service.graph_bridge import ndb_to_geometric_graph

# Construir espaço geométrico
space, node_ids, id_to_idx = ndb_to_geometric_graph(client, collection)

# Treinar modelo de incerteza
estimator = EpistemicUncertaintyEstimator(space, node_ids)
observed = [(id_to_idx[n.id], n.energy) for n in nodes if n.energy > 0.1]
obs_idx = np.array([o[0] for o in observed])
obs_val = np.array([o[1] for o in observed])
estimator.fit(obs_idx, obs_val)

# Encontrar lacunas — usar como seed para hipóteses
gaps = estimator.find_knowledge_gaps(top_k=10)
# Passar gaps para hypothesis_generator como contexto adicional
```

---

### Fase 4 — Kernels Hiperbólicos Exatos (Semana 8-9)

**Objetivo**: Usar o espaço Hiperbólico da GeometricKernels para validar e melhorar
operações do `nietzsche-hyp-ops`.

#### 4.1 Hyperbolic Kernel Service

```python
from geometric_kernels.spaces import Hyperbolic
from geometric_kernels.kernels import MaternGeometricKernel
import numpy as np

class HyperbolicKernelService:
    """
    Kernels no espaço hiperbólico para embeddings Poincaré do NietzscheDB.

    NOTA: GeometricKernels usa modelo hiperbolóide internamente.
    NietzscheDB usa Poincaré ball. Conversão necessária.
    """

    def __init__(self, dim: int = 128):
        # dim do espaço hiperbólico (mesma dim dos PoincareVector no NietzscheDB)
        self.space = Hyperbolic(dim=dim)
        self.kernel = MaternGeometricKernel(self.space)
        self.params = self.kernel.init_params()

    def poincare_to_hyperboloid(self, poincare_coords: np.ndarray) -> np.ndarray:
        """
        Converte coordenadas Poincaré ball → hyperboloid model.

        Poincaré ball: x ∈ R^d, ||x|| < 1
        Hyperboloid: y ∈ R^{d+1}, y₀² - y₁² - ... - y_d² = 1, y₀ > 0

        Fórmula: y₀ = (1 + ||x||²) / (1 - ||x||²)
                  yᵢ = 2xᵢ / (1 - ||x||²)    para i = 1..d
        """
        norm_sq = np.sum(poincare_coords ** 2, axis=-1, keepdims=True)
        norm_sq = np.clip(norm_sq, 0, 1 - 1e-7)  # evitar divisão por zero

        scale = 1.0 / (1.0 - norm_sq)
        y0 = (1.0 + norm_sq) * scale
        yi = 2.0 * poincare_coords * scale

        return np.concatenate([y0, yi], axis=-1)

    def compute_similarity(self, poincare_points: np.ndarray,
                           nu: float = 2.5, lengthscale: float = 1.0) -> np.ndarray:
        """
        Calcula matriz de similaridade Matérn no espaço hiperbólico.

        Usa feature maps (rejection sampling) para espaços não-compactos.
        Respeita curvatura — NÃO é distância euclidiana.
        """
        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([lengthscale])

        X = self.poincare_to_hyperboloid(poincare_points)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def kernel_knn(self, query_poincare: np.ndarray,
                   candidates_poincare: np.ndarray,
                   k: int = 10, nu: float = 2.5) -> list[tuple[int, float]]:
        """
        KNN baseado em kernel hiperbólico em vez de distância Poincaré.

        Vantagem sobre poincare_distance:
        - Matérn kernel captura locality multi-escala
        - ν controla suavidade (0.5=rough, 2.5=smooth, ∞=heat)
        - Pode ser mais robusto para nós perto da borda do disco
        """
        query_hyp = self.poincare_to_hyperboloid(query_poincare.reshape(1, -1))
        cands_hyp = self.poincare_to_hyperboloid(candidates_poincare)

        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([1.0])

        K = self.kernel.K(self.params, cands_hyp, query_hyp)
        similarities = np.array(K).flatten()

        top_k_idx = np.argsort(-similarities)[:k]
        return [(int(idx), float(similarities[idx])) for idx in top_k_idx]
```

#### 4.2 Validação de GAUSS_KERNEL NQL

O NQL `GAUSS_KERNEL(n, t)` actual usa:

```rust
h_t(x) = exp(−‖x‖² / (4t))  // Euclidiano! Não respeita curvatura
```

Corrigir para usar distância hiperbólica:

```rust
// Em nietzsche-query/src/executor.rs, MathFunc::GaussKernel:
// ANTES: d_sq = ||x||² (euclidiano)
// DEPOIS: d_sq = acosh(1 + 2||x||²/(1-||x||²))² (Poincaré desde a origem)
```

A GeometricKernels serve como referência para validar a correção.

---

### Fase 5 — Export ONNX para Runtime Rust (Semana 10-12)

**Objetivo**: Exportar feature maps treinados para ONNX, executar em Rust via `nietzsche-neural`.

#### 5.1 Export Pipeline (`export_onnx.py`)

```python
import torch
import numpy as np
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.feature_maps import default_feature_map

class GeometricFeatureMapExporter:
    """
    Exporta feature maps de GeometricKernels para ONNX.

    Feature map: φ(x) ∈ R^M tal que k(x,y) ≈ ⟨φ(x), φ(y)⟩

    No Rust, basta:
    1. Computar φ(x) via ONNX
    2. Similaridade = dot product dos features
    """

    def __init__(self, space, num_features: int = 256):
        self.space = space
        self.num_features = num_features
        self.kernel = MaternGeometricKernel(space)
        self.feature_map = default_feature_map(space=space, num=num_features)

    def export_graph_features(self, output_path: str,
                               nu: float = 2.5, lengthscale: float = 1.0):
        """
        Para Graph spaces: exporta eigenvectors + spectrum weights como ONNX.

        O modelo ONNX recebe: node_index (int) → feature_vector (float[M])
        """
        params = self.kernel.init_params()
        params["nu"] = np.array([nu])
        params["lengthscale"] = np.array([lengthscale])

        # Pré-computar features para todos os nós
        n_nodes = self.space.num_vertices
        all_indices = np.arange(n_nodes).reshape(-1, 1)

        key = np.random.RandomState(42)
        _, features = self.feature_map(all_indices, params, key=key)
        features = np.array(features)  # shape: (n_nodes, num_features)

        # Criar modelo PyTorch wrapper para export
        class FeatureLookup(torch.nn.Module):
            def __init__(self, feature_table):
                super().__init__()
                self.features = torch.nn.Embedding.from_pretrained(
                    torch.tensor(feature_table, dtype=torch.float32),
                    freeze=True
                )

            def forward(self, node_idx):
                return self.features(node_idx)

        model = FeatureLookup(features)
        dummy_input = torch.tensor([0], dtype=torch.long)

        torch.onnx.export(
            model, dummy_input, output_path,
            input_names=["node_index"],
            output_names=["feature_vector"],
            dynamic_axes={"node_index": {0: "batch_size"}},
            opset_version=17,
        )

    def export_hyperbolic_features(self, output_path: str,
                                     nu: float = 2.5, lengthscale: float = 1.0):
        """
        Para Hyperbolic spaces: exporta feature map contínuo como ONNX.

        O modelo ONNX recebe: hyperboloid_coords (float[d+1]) → features (float[M])

        NOTA: Requer pré-computar rejection sampling frequencies.
        Exporta lookup table de frequências + power function computation.
        """
        # Hyperbolic feature maps são mais complexos (rejection sampling)
        # Estratégia: pré-amostrar frequências, exportar como constantes
        params = self.kernel.init_params()
        params["nu"] = np.array([nu])
        params["lengthscale"] = np.array([lengthscale])

        # Gerar pontos de teste para capturar as frequências
        key = np.random.RandomState(42)
        test_points = self.space.random(key, 100)
        _, test_features = self.feature_map(test_points, params, key=key)

        # Para hyperbolic: feature map depende das frequências amostradas
        # Exportar como modelo com frequências fixas
        # (implementação detalhada depende da versão da biblioteca)
        pass
```

#### 5.2 Integração com `nietzsche-neural`

```rust
// Em nietzsche-neural/src/lib.rs, adicionar:

impl ModelRegistry {
    /// Carrega feature map ONNX exportado de GeometricKernels.
    ///
    /// Uso: similarity = dot(feature(node_a), feature(node_b))
    pub fn geometric_similarity(
        &self,
        model_name: &str,
        node_idx_a: usize,
        node_idx_b: usize,
    ) -> Result<f32> {
        let feat_a = self.infer_f32(
            model_name,
            vec![1],
            vec![node_idx_a as f32],
        )?;
        let feat_b = self.infer_f32(
            model_name,
            vec![1],
            vec![node_idx_b as f32],
        )?;

        // Dot product = kernel approximation
        let sim: f32 = feat_a.iter()
            .zip(feat_b.iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(sim)
    }
}
```

---

### Fase 6 — NQL Extensions (Semana 13-14)

**Objetivo**: Novas funções NQL que expõem os kernels geométricos.

#### 6.1 Novas MathFunc no AST

```rust
// Em nietzsche-query/src/ast.rs, adicionar ao enum MathFunc:

/// Kernel Matérn geométrico (via ONNX feature map)
MaternKernel,          // MATERN_KERNEL(n1, n2, nu, lengthscale)

/// Incerteza epistémica (variância GP)
EpistemicUncertainty,  // EPISTEMIC_UNCERTAINTY(n)

/// Heat kernel hiperbólico corrigido (usa distância Poincaré)
HyperbolicHeatKernel,  // HYPERBOLIC_HEAT(n, t)
```

#### 6.2 Exemplos NQL

```sql
-- Buscar conceitos mais similares via Matérn kernel
MATCH (n:Semantic)
WHERE MATERN_KERNEL(n, $target_node, 2.5, 1.0) > 0.5
ORDER BY MATERN_KERNEL(n, $target_node, 2.5, 1.0) DESC
LIMIT 20

-- Encontrar lacunas de conhecimento
MATCH (n:Concept)
WHERE EPISTEMIC_UNCERTAINTY(n) > 0.8
ORDER BY EPISTEMIC_UNCERTAINTY(n) DESC
LIMIT 10

-- Heat kernel hiperbólico correto (não euclidiano)
MATCH (n:Semantic)
WHERE HYPERBOLIC_HEAT(n, 1.0) > 0.3
ORDER BY HYPERBOLIC_HEAT(n, 1.0) DESC
```

---

### Fase 7 — Agency Integration (Semana 15-16)

**Objetivo**: Usar incerteza geométrica no loop de agência (reactor + evolution).

#### 7.1 Novo Intent

```rust
// Em nietzsche-agency/src/reactor.rs, enum AgencyIntent:

/// Resultado de análise de incerteza geométrica
GeometricUncertainty {
    node_id: Uuid,
    variance: f32,         // variância GP
    suggested_action: UncertaintyAction,
}

pub enum UncertaintyAction {
    Research,        // alta incerteza → pesquisar mais
    Consolidate,     // baixa incerteza → consolidar com vizinhos
    Prune,           // zero incerteza + zero energia → candidato a remoção
}
```

#### 7.2 Evolution27 com Incerteza

```rust
// Em evolution_27.rs, usar incerteza para guiar mutações:

// Nós com alta incerteza → candidatos a NEW_EDGE (conectar a conceitos estáveis)
// Nós com alta incerteza + alta energia → candidatos a NEW_CONCEPT (expandir)
// Nós com baixa incerteza + baixa energia → candidatos a REMOVE_EDGE (simplificar)
```

---

## Cronograma Resumido

| Fase | Semana | Entrega | Dependência |
|------|--------|---------|-------------|
| **1. Fundação Python** | 1-2 | `graph_bridge.py`, `kernel_service.py`, testes básicos | GeometricKernels instalado |
| **2. Calibração Chebyshev** | 3-4 | Relatório de erro, K_max ótimo por escala | Fase 1 + NietzscheDB server |
| **3. Motor de Incerteza** | 5-7 | `uncertainty.py`, integração NietzscheLab | Fase 1 + GPyTorch |
| **4. Kernels Hiperbólicos** | 8-9 | `HyperbolicKernelService`, fix GAUSS_KERNEL NQL | Fase 1 |
| **5. Export ONNX** | 10-12 | Feature maps ONNX, `geometric_similarity()` em Rust | Fases 1+4 |
| **6. NQL Extensions** | 13-14 | `MATERN_KERNEL`, `EPISTEMIC_UNCERTAINTY`, `HYPERBOLIC_HEAT` | Fase 5 |
| **7. Agency Integration** | 15-16 | `GeometricUncertainty` intent, Evolution27 guiado | Fases 3+6 |

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Eigendecomposition lenta em grafos grandes (>10K nós) | Alto | Truncar a top-K eigenvalues; usar ARPACK sparse; subgraph sampling |
| Feature map ONNX perde precisão vs kernel exato | Médio | Benchmark antes/depois; aumentar `num_features` se necessário |
| GPyTorch memory em grafos com muitos nós observados | Médio | Usar inducing points (SVGP) em vez de ExactGP para N>2000 |
| Conversão Poincaré→hyperboloid com norma ≈ 1.0 | Alto | Clamp norm a 1-ε antes de converter; monitorar gradientes |
| GeometricKernels depende de NumPy >=2.0 | Baixo | Fixar versão em requirements.txt |

---

## Métricas de Sucesso

### Fase 2 (Calibração)
- [ ] Overlap@20 entre Chebyshev e kernel exato > 85% para t=0.1 e t=1.0
- [ ] Identificar K_max ótimo para cada escala cognitiva

### Fase 3 (Incerteza)
- [ ] GP identifica ≥ 70% das lacunas reais (validação manual em 3 collections)
- [ ] Tempo de treino GP < 30s para subgraph de 2000 nós

### Fase 5 (ONNX)
- [ ] Feature map ONNX reproduz kernel com correlação > 0.95
- [ ] Inference time < 1ms por par de nós no Rust

### Fase 7 (Agency)
- [ ] Evolution27 com incerteza produz mutações de maior qualidade (delta score > baseline)
- [ ] EpistemologyDaemon prioriza regiões de alta incerteza

---

## Decisões Técnicas

### ❌ NÃO portar GeometricKernels para Rust
- Custo muito alto (NumPy/SciPy/PyTorch ecosystem inteiro)
- Melhor abordagem: Python treina → ONNX → Rust infere

### ✅ Manter Chebyshev como fast-path
- GeometricKernels = ground truth (offline, calibração)
- Chebyshev = runtime (online, tempo real)
- Não substituir, complementar

### ✅ Usar Graph space (não Hyperbolic) como ponto de partida
- Graph space usa Laplacian eigendecomposition — análogo direto ao nietzsche-pregel
- Hyperbolic space requer feature maps (mais complexo, Fase 4)

### ✅ GPyTorch como frontend GP (não GPJax)
- PyTorch é mais maduro para ONNX export
- GPyTorch tem inducing points nativos (escala para grafos grandes)

### ⚠️ Binary Quantization permanece REJEITADA
- GeometricKernels NÃO altera esta decisão
- Kernels geométricos dependem de magnitude (distância geodésica)
- sign(x) continua a destruir hierarquia hiperbólica
