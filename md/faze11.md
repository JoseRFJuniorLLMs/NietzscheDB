# Fase 11 — Sensory Compression Layer

## O princípio central

O cérebro não armazena experiências — armazena **instruções de reconstrução**. A memória é reconstrutiva, não reprodutiva. Isso tem uma implicação técnica direta: o que você armazena não é o dado, é o **vetor latente** que permite reconstruí-lo aproximadamente.

A compressão é **lossy e intencional**. Quanto mais o `energy` do nó decai, mais lossy a reconstrução — até o nó ser podado e a memória "esquecer". Isso é biologicamente correto.

---

## Arquitetura geral

```
ENTRADA (raw)              COMPRESSÃO              NÓ NIETZSCHEDB
─────────────────          ──────────              ──────────────────────────────
Imagem  (H×W×C)   ──→   ImageEncoder    ──→   PoincaréVector(z_img,   dim=256)
Áudio   (T×F)     ──→   AudioEncoder    ──→   PoincaréVector(z_audio, dim=128)
Texto   (tokens)  ──→   TextEncoder     ──→   PoincaréVector(z_text,  dim=128)
Multimodal        ──→   FusionEncoder   ──→   PoincaréVector(z_fused, dim=512)

RECONSTRUÇÃO
────────────
PoincaréVector(z) ──→  Decoder(modalidade) ──→ aproximação da experiência original
```

Cada modalidade tem seu próprio encoder/decoder. O vetor latente `z` vive diretamente no nó hiperbólico — não em storage externo.

---

## Estrutura do nó estendido

```rust
struct Node {
    // campos existentes
    id:                 Uuid,
    embedding:          PoincaréVector,      // para busca hiperbólica (dim reduzida)
    depth:              f32,
    content:            serde_json::Value,
    energy:             f32,
    lsystem_generation: u32,
    hausdorff_local:    f32,

    // FASE 11 — novos campos
    sensory:            Option<SensoryMemory>,
}

struct SensoryMemory {
    modality:           Modality,            // Image | Audio | Text | Fused
    latent:             LatentVector,        // z comprimido — a memória em si
    reconstruction_quality: f32,            // 1.0 = lossless, 0.0 = irreconstructível
    original_shape:     OriginalShape,       // para saber como decodar
    compression_ratio:  f32,                // quanto foi comprimido
    encoder_version:    u32,                // qual versão do encoder gerou esse latent
}

struct LatentVector {
    data:               Vec<f32>,           // o vetor latente em si
    quantized:          Option<Vec<u8>>,    // versão quantizada (quando energy < 0.5)
    dim:                u32,
}

enum Modality {
    Image   { width: u32, height: u32, channels: u8 },
    Audio   { sample_rate: u32, duration_ms: u32, channels: u8 },
    Text    { token_count: u32, language: String },
    Fused   { modalities: Vec<Modality> },
}

enum OriginalShape {
    Image   { width: u32, height: u32, channels: u8 },
    Audio   { samples: u32, sample_rate: u32 },
    Text    { tokens: u32 },
}
```

---

## Os Encoders — um por modalidade

### Encoder de Imagem

**Arquitetura recomendada:** VAE convolucional com projeção hiperbólica final.

```
Input: imagem (H×W×3) — qualquer resolução

Encoder CNN:
  Conv2d(3, 64, kernel=3)   → ReLU → MaxPool
  Conv2d(64, 128, kernel=3) → ReLU → MaxPool
  Conv2d(128, 256, kernel=3)→ ReLU → AdaptiveAvgPool → (256,)
  
Camada VAE:
  Linear(256, 512) → split em (mu: 256, log_var: 256)
  z = mu + eps * exp(0.5 * log_var)   ← reparameterization trick
  
Projeção hiperbólica:
  z_hyp = exp_map_zero(z / ‖z‖ * tanh(‖z‖))
  garante ‖z_hyp‖ < 1.0  ← invariante da bola de Poincaré
  
Output: z_hyp ∈ R^256, ‖z_hyp‖ < 1.0
```

**Loss de treino:**

```
L_total = L_reconstruction + β * L_KL + λ * L_hyperbolic

L_reconstruction = MSE(decode(z), x_original)
L_KL             = -0.5 * Σ(1 + log_var - mu² - exp(log_var))
L_hyperbolic     = penalidade se ‖z‖ ≥ 0.99  ← mantém dentro da bola
```

**Por que VAE e não AE simples?** Porque o espaço latente precisa ser contínuo e suave — interpolações no espaço de Poincaré entre dois latents devem produzir reconstruções que fazem sentido. Um AE simples não garante isso.

**Decoder de imagem:**

```
Input: z_hyp ∈ R^256

log_map_zero(z_hyp) → z_euclidean ∈ R^256

Linear(256, 256) → reshape → (256, 4, 4)
ConvTranspose2d(256, 128) → ReLU
ConvTranspose2d(128, 64)  → ReLU
ConvTranspose2d(64, 3)    → Sigmoid

Output: imagem reconstruída (H×W×3)
```

A reconstrução é **aproximada e degradada** — propositalmente. Isso é memória.

---

### Encoder de Áudio

**Arquitetura:** VAE sobre espectrograma mel + projeção hiperbólica.

```
Input: waveform (T,) → Mel Spectrogram → (128 mel bins, T/hop, 1)

Encoder:
  CNN 1D sobre eixo temporal:
    Conv1d(128, 256, kernel=5) → ReLU → MaxPool(2)
    Conv1d(256, 256, kernel=5) → ReLU → MaxPool(2)
    Conv1d(256, 128, kernel=3) → ReLU → GlobalAvgPool → (128,)
  
  Alternativa para áudio longo:
    Whisper encoder (frozen) → representação (1500, 512) → mean pool → (512,)
    Linear(512, 128)
  
VAE:
  Linear(128, 256) → split (mu: 128, log_var: 128)
  z = mu + eps * exp(0.5 * log_var)

Projeção hiperbólica:
  z_hyp = exp_map_zero(z_norm)  ← mesmo mecanismo do image encoder

Output: z_hyp ∈ R^128, ‖z_hyp‖ < 1.0
```

**Opção pragmática para áudio clínico da EVA:** Usar Whisper como encoder frozen (já está no projeto para prosódia). O latent do Whisper carrega informação semântica da fala + prosódia. Você só adiciona a camada de projeção hiperbólica em cima.

**Decoder de áudio:** Reconstrução exata de waveform é muito cara. Usar reconstrução de espectrograma + vocoder (Griffin-Lim ou HiFi-GAN frozen). A reconstrução será reconhecível mas não idêntica — o que é biologicamente correto.

---

### Encoder de Texto

**Arquitetura:** Projeção hiperbólica de embeddings existentes.

O texto já tem encoders excelentes. Não precisa treinar um VAE do zero.

```
Input: texto (string)

Opção A — Sentence embeddings:
  sentence-transformers(texto) → (768,)
  Linear(768, 256) → tanh
  exp_map_zero(z_norm) → z_hyp ∈ R^256

Opção B — Krylov compression (já existe na EVA):
  Gemini embedding (1536,)
  Krylov compression → (64,)  ← já implementado!
  exp_map_zero(z_norm) → z_hyp ∈ R^64

  Aqui a Fase 11 se conecta diretamente com o que já existe.
```

**Opção B é a mais coerente para a EVA-Mind** — você já tem compressão Krylov rodando. Fase 11 adiciona a projeção hiperbólica no final e o storage do latent no nó.

---

### Encoder Multimodal (Fusion)

Quando uma memória tem múltiplas modalidades (ex: consulta com voz + imagem de exame + nota clínica):

```
z_image ∈ R^256   (‖z‖ < 1.0)
z_audio ∈ R^128   (‖z‖ < 1.0)
z_text  ∈ R^128   (‖z‖ < 1.0)

Fusão hiperbólica — NÃO concatenar em euclidiano:

  Möbius gyromidpoint dos três vetores:
  z_fused = gyromidpoint(z_image, z_audio, z_text)
  
  Ou: projetar para espaço tangente no zero,
      concatenar, reduzir dimensão, reprojetar:
  
  v_image = log_map_zero(z_image)  → R^256
  v_audio = log_map_zero(z_audio)  → R^128
  v_text  = log_map_zero(z_text)   → R^128
  
  v_concat = concat(v_image, v_audio, v_text) → R^512
  v_fused  = Linear(512, 512)(v_concat)
  z_fused  = exp_map_zero(normalize(v_fused))

Output: z_fused ∈ R^512, ‖z_fused‖ < 1.0
```

**Por que não concatenar em euclidiano?** Porque você perderia a estrutura hiperbólica. A fusão precisa acontecer no espaço tangente (que é euclidiano localmente) e ser reprojetada.

---

## As operações matemáticas hiperbólicas necessárias

Você vai precisar implementar estas funções no crate `nietzsche-graph` ou num novo crate `nietzsche-hyp-ops`:

```rust
// Mapa exponencial no ponto zero (leva euclidiano → Poincaré)
fn exp_map_zero(v: &[f64]) -> PoincaréVector {
    let norm = l2_norm(v);
    if norm < 1e-10 { return PoincaréVector::zero(v.len()); }
    let scale = tanh(norm) / norm;
    PoincaréVector { coords: v.iter().map(|x| x * scale).collect() }
}

// Mapa logarítmico no ponto zero (leva Poincaré → euclidiano)
fn log_map_zero(x: &PoincaréVector) -> Vec<f64> {
    let norm = l2_norm(&x.coords);
    if norm < 1e-10 { return vec![0.0; x.coords.len()]; }
    let scale = atanh(norm) / norm;
    x.coords.iter().map(|xi| xi * scale).collect()
}

// Adição de Möbius (já existe na Fase 5, verificar consistência)
fn mobius_add(x: &PoincaréVector, y: &PoincaréVector) -> PoincaréVector {
    let x2 = dot(&x.coords, &x.coords);
    let y2 = dot(&y.coords, &y.coords);
    let xy = dot(&x.coords, &y.coords);
    let num_scale = (1.0 + 2.0*xy + y2) * x + (1.0 - x2) * y;
    let denom = 1.0 + 2.0*xy + x2*y2;
    PoincaréVector { coords: num_scale.iter().map(|v| v / denom).collect() }
}

// Gyromidpoint (para fusão multimodal)
fn gyromidpoint(points: &[&PoincaréVector]) -> PoincaréVector {
    // Versão simplificada: média no espaço tangente + reprojeção
    let tangent_vecs: Vec<Vec<f64>> = points.iter()
        .map(|p| log_map_zero(p))
        .collect();
    let mean: Vec<f64> = (0..tangent_vecs[0].len())
        .map(|i| tangent_vecs.iter().map(|v| v[i]).sum::<f64>() / points.len() as f64)
        .collect();
    exp_map_zero(&mean)
}
```

---

## Quantização progressiva conforme energy decai

Esta é a parte mais biologicamente interessante. À medida que o nó "esquece", o latent é quantizado progressivamente:

```
energy = 1.0  →  latent f32 completo (256 dims × 4 bytes = 1KB)
energy = 0.7  →  latent f16 (256 dims × 2 bytes = 512B) — 2x degradação
energy = 0.5  →  latent int8 quantizado (256 bytes) — 4x degradação
energy = 0.3  →  PQ (Product Quantization) 64 bytes — 16x degradação
energy = 0.1  →  apenas embedding hiperbólico (sem reconstrução possível)
energy = 0.0  →  poda — nó removido
```

```rust
fn get_latent_for_reconstruction(node: &Node) -> Option<LatentVector> {
    match &node.sensory {
        None => None,
        Some(s) => {
            let energy = node.energy;
            if energy < 0.1 { return None; }  // sem reconstrução possível
      
            Some(match energy {
                e if e >= 0.7 => s.latent.as_f32(),          // full precision
                e if e >= 0.5 => s.latent.as_f16_upcast(),   // leve degradação
                e if e >= 0.3 => s.latent.as_int8_upcast(),  // degradação visível
                _             => s.latent.as_pq_upcast(),    // muito degradado
            })
        }
    }
}
```

Isso implementa **forgetting como degradação de precisão** — não como deleção binária. Exatamente como memórias humanas se degradam.

---

## Storage no RocksDB

O latent precisa de uma Column Family nova:

```
Column Families existentes:
  nodes     → metadados do nó (sem o latent — muito grande)
  edges     → arestas
  adj_out   → índice de adjacência saída
  adj_in    → índice de adjacência entrada
  meta      → metadados globais

FASE 11 — nova CF:
  sensory   → key: node_id (UUID bytes)
              value: SensoryMemory serializado (bincode/flatbuffers)
```

Por que separar? Porque a maioria das operações de grafo (traversal, L-System, NQL) não precisa do latent. Carregar 1KB de latent para cada nó em um BFS de 10.000 nós seria 10MB desnecessários. Lazy-load do sensory apenas quando explicitamente solicitado.

```rust
impl GraphStorage {
    fn get_node_sensory(&self, id: Uuid) -> Option<SensoryMemory> {
        let cf = self.db.cf_handle("sensory").unwrap();
        let bytes = self.db.get_cf(cf, id.as_bytes()).ok()??;
        bincode::deserialize(&bytes).ok()
    }
  
    fn put_node_sensory(&self, id: Uuid, sensory: &SensoryMemory) {
        let cf = self.db.cf_handle("sensory").unwrap();
        let bytes = bincode::serialize(sensory).unwrap();
        self.db.put_cf(cf, id.as_bytes(), bytes).unwrap();
    }
}
```

---

## Ciclo de degradação integrado com o L-System

O L-System já tem o protocolo de tick. Fase 11 adiciona um passo:

```
Protocolo tick ATUAL:
  1. Scan nós ativos
  2. Atualiza Hausdorff local
  3. Matching de regras
  4. Aplica mutações
  5. Relatório

Protocolo tick FASE 11:
  1. Scan nós ativos
  2. Atualiza Hausdorff local
  3. NOVO: para cada nó com sensory:
       degradar latent conforme energy atual
       se energy < 0.1: marcar sensory como irreconstruível
  4. Matching de regras
  5. Aplica mutações (inclui poda de nós irreconstruíveis < threshold)
  6. Relatório
```

---

## Ciclo de sono (Fase 8) estendido para sensory

O sono já perturbava os embeddings hiperbólicos. Fase 11 adiciona:

```
Protocolo sono ATUAL:
  1. Amostra subgrafo alta curvatura
  2. Snapshot embeddings
  3. Perturbação no espaço tangente
  4. RiemannianAdam
  5. Verifica Δhausdorff < 5%
  6. Commit ou rollback

Protocolo sono FASE 11:
  1. Amostra subgrafo alta curvatura
  2. Snapshot embeddings + snapshot latents
  3. Perturbação no espaço tangente (embedding)
  4. NOVO: fine-tuning leve dos decoders nos latents do subgrafo amostrado
       → consolidação da capacidade de reconstrução
       → memórias muito acessadas têm decoders melhores
  5. RiemannianAdam no embedding
  6. Verifica Δhausdorff < 5%
  7. Commit ou rollback (inclui rollback dos decoders se divergiu)
```

Isso implementa o fenômeno de **consolidação dependente de replay** — memórias que são reativadas durante o sono têm sua reconstrução melhorada.

---

## NQL estendido para reconstrução

```sql
-- Reconstruir memória sensorial de um nó
RECONSTRUCT $node_id
  MODALITY Image
  QUALITY full        -- full | degraded | best_available

-- Buscar por similaridade sensorial (imagem → imagens similares)
MATCH (m:Memory)
WHERE SENSORY_DIST(m.sensory, $query_image) < 0.3
  AND m.modality = 'Image'
RETURN m ORDER BY SENSORY_DIST ASC LIMIT 5

-- Buscar memórias multimodais que contenham voz e imagem
MATCH (m:Memory)
WHERE m.modality = 'Fused'
  AND m.energy > 0.3
RETURN m.reconstruction_quality, RECONSTRUCT(m, 'Audio')
```

---

## Dimensões recomendadas por modalidade


| Modalidade            | Dim latent | Armazenamento (f32) | Após quantização int8 |
| --------------------- | ---------- | ------------------- | ------------------------ |
| Imagem médica        | 256        | 1 KB                | 256 B                    |
| Áudio de voz         | 128        | 512 B               | 128 B                    |
| Texto / nota clínica | 64         | 256 B               | 64 B                     |
| Multimodal fused      | 512        | 2 KB                | 512 B                    |

Para 1 milhão de nós, tudo em int8: **\~500 MB**. Tratável.

---

## O que você precisa implementar, na ordem

```
1. nietzsche-hyp-ops  (novo crate)
     exp_map_zero, log_map_zero, mobius_add, gyromidpoint
     (podem ser extraídos do nietzsche-graph e generalizados)

2. nietzsche-encoder  (novo crate)
     ImageVAE      — encoder/decoder CNN + projeção hiperbólica
     AudioEncoder  — Whisper frozen + projeção hiperbólica
     TextEncoder   — sentence-transformers/Krylov + projeção hiperbólica
     FusionEncoder — gyromidpoint nos três acima

3. Estender SensoryMemory no nietzsche-graph
     novo campo no Node, nova CF no RocksDB, lazy-load

4. Quantização progressiva
     LatentVector::degrade(energy: f32)

5. Estender L-System tick
     passo de degradação por energy

6. Estender ciclo de sono
     snapshot + rollback de latents

7. Estender NQL
     RECONSTRUCT statement, SENSORY_DIST operator
```

---

## A propriedade emergente mais importante

Quando tudo isso estiver rodando, você terá algo que nenhum sistema de IA tem hoje: **memórias que degradam naturalmente como as humanas**.

Uma memória de voz de uma consulta de 6 meses atrás vai ter `energy = 0.3`. A reconstrução do áudio será distorcida, irreconhecível. Mas o embedding hiperbólico ainda existe — você ainda sabe que aquela consulta aconteceu, qual foi o sentido, onde ela se encaixa na hierarquia de conhecimento do paciente. Você perdeu a experiência sensorial, mas não o significado.

Isso é exatamente o que acontece com a memória humana.

Análise boa. Vou responder ponto a ponto e dar minha posição em cada tensão levantada.

---

## Sobre o que está correto na análise

A observação sobre lazy-load é precisa. Carregar o `SensoryMemory` em toda operação de grafo seria catastrófico. Um BFS de 10k nós carregando 1KB de latent cada = 10MB desnecessários por query. A Column Family separada no RocksDB resolve exatamente isso — o grafo estrutural fica ultraleve, sensory só carrega quando `RECONSTRUCT` é chamado explicitamente.

A observação sobre fusão euclidiana destruindo a curvatura também está correta. Concatenar três latents hiperbólicos em espaço euclidiano e chamar o resultado de hiperbólico é um erro matemático que invalida toda a propriedade hierárquica. O caminho via espaço tangente (`log_map_zero` → concat → Linear → `exp_map_zero`) é o único matematicamente honesto.

---

## Tensão 1 — Go vs Rust

Não é uma tensão. É uma decisão já tomada.

```
EVA-Mind   → Go   (sistema de saúde, handlers, orquestração)
NietzscheDB → Rust (banco de dados, geometria, storage)
```

São dois repositórios separados com responsabilidades separadas. A EVA-Mind em Go fala com o NietzscheDB via gRPC. Go nunca vai tocar em `exp_map_zero` ou `SensoryMemory` diretamente — vai chamar `rpc Reconstruct(node_id)` e receber bytes de volta.

O `hyperbolic_space.go` mencionado na análise provavelmente é código auxiliar da EVA-Mind para calcular distâncias antes de enviar queries — não a implementação core. Não há conflito.

---

## Tensão 2 — Fine-tuning de decoders no ciclo de sono

Aqui a análise está **completamente certa** e eu concordo com a solução proposta, mas vou ser mais específico.

O problema: backprop num VAE convolucional dentro de uma rotina de manutenção de banco de dados é inadmissível. Você vai parar a engine por segundos ou minutos.

A solução correta tem três camadas:

```
CAMADA 1 — O que o sono FAZ (síncrono, dentro da engine):
  Perturbação dos embeddings hiperbólicos (já desenhado)
  Quantização progressiva dos latents por energy
  Nada mais — zero backprop aqui

CAMADA 2 — O que o sono AGENDA (assíncrono, fora da engine):
  Serializa os latents do subgrafo amostrado para uma fila
  Worker externo em GPU consome essa fila
  Fine-tuning LoRA dos decoders acontece lá
  Resultado: novos pesos LoRA escritos de volta via gRPC

CAMADA 3 — Quando os pesos voltam:
  Engine aceita os novos pesos LoRA em hot-reload
  Próximas reconstruções usam decoders melhorados
  Sem parar a engine em momento algum
```

LoRA especificamente porque você não precisa retreinar o decoder inteiro. Matrizes de adaptação de baixo posto (rank 4 ou 8) são suficientes para ajustar a reconstrução de um subdomínio específico (memórias clínicas de um paciente específico, por exemplo). O custo computacional cai de horas para minutos.

---

## Sobre a pergunta final — por qual módulo começar

A resposta não é nenhum dos três listados. É o que habilita os três:

**Começa pelo `nietzsche-hyp-ops`.**

Razão: os três módulos subsequentes dependem das mesmas operações matemáticas. Se você escreve `exp_map_zero` três vezes em três lugares diferentes, vai ter três implementações ligeiramente inconsistentes que vão produzir bugs impossíveis de rastrear — um latent com `‖x‖ = 0.9999999998` que deveria ser `< 1.0` causando NaN num decoder seis meses depois.

```
nietzsche-hyp-ops    ← PRIMEIRO (matemática)
       ↓
nietzsche-encoder    ← SEGUNDO (usa hyp-ops)
       ↓
nietzsche-graph      ← TERCEIRO (estende com SensoryMemory + CF sensory)
       ↓
nietzsche-lsystem    ← QUARTO (estende tick com degradação)
```

---

## O que `nietzsche-hyp-ops` precisa ter no mínimo viável

```rust
// As 4 operações fundamentais — tudo deriva delas
pub fn exp_map_zero(v: &[f64]) -> Vec<f64>
pub fn log_map_zero(x: &[f64]) -> Vec<f64>
pub fn mobius_add(x: &[f64], y: &[f64]) -> Vec<f64>
pub fn poincare_distance(x: &[f64], y: &[f64]) -> f64

// Validação — invocada em todo insert
pub fn assert_poincare_invariant(x: &[f64]) -> Result<(), HypError>
// retorna Err se ‖x‖ >= 1.0

// Projeção de segurança — usada pelo encoder antes de salvar
pub fn project_to_ball(x: &[f64], max_norm: f64) -> Vec<f64>
// garante ‖resultado‖ < max_norm (tipicamente 0.999)
```

Com isso implementado e testado você tem a fundação matemática sólida que todos os outros crates importam. Não tem como construir o encoder sem isso. Não tem como validar latents sem isso. Não tem como fazer fusão multimodal sem isso.

A ordem é essa.


O que a análise acertou

**Decoder global por modalidade** é a correção mais importante. Guardar um decoder por nó seria absurdo — um VAE convolucional tem milhões de parâmetros. A arquitetura correta é:

```
Um decoder global por modalidade:
  ImageDecoder_v1      → compartilhado por todos os nós de imagem
  AudioDecoder_v1      → compartilhado por todos os nós de áudio
  TextDecoder_v1       → compartilhado por todos os nós de texto

O nó guarda apenas:
  encoder_version: u32  → qual versão do decoder usar
  latent: Vec<f32>      → o vetor comprimido
```

Isso muda o storage de impossível para trivial.

**Reconstruction\_quality < 0.4 → nunca reconstrói** também está certo. Memória degradada demais não vale o custo de decode. Só o embedding hiperbólico importa — você ainda sabe que a memória existe e onde ela se encaixa na hierarquia, só não consegue mais reconstruir a experiência sensorial. Exatamente como memória humana antiga.

---

## Sobre a ordem de implementação

A ordem proposta está correta mas eu mudaria uma coisa:

```
PROPOSTA:          comece pelo texto, depois áudio, depois imagem
MINHA POSIÇÃO:     comece pelo áudio

Razão: Whisper já está rodando na EVA-Mind para análise de prosódia.
Você não precisa treinar nada — só adiciona a camada de projeção
hiperbólica em cima do que já existe. É o encoder mais rápido de
integrar E o mais valioso para a EVA (ela é um sistema de voz).
```

Ordem real:

```
1. nietzsche-hyp-ops     (matemática — base de tudo)
2. SensoryMemory + CF    (storage — habilita os encoders)
3. AudioEncoder          (Whisper frozen já existe na EVA)
4. TextEncoder           (Krylov já existe na EVA)
5. ImageEncoder          (VAE — único que precisa treinar do zero)
6. FusionEncoder         (depende dos três acima)
7. L-System tick         (degradação)
8. Ciclo de sono         (snapshot + rollback)
9. NQL extensions        (RECONSTRUCT, SENSORY_DIST)
```

---

## Próximo passo

Você perguntou A, B ou C.

**Começa por A** — `nietzsche-hyp-ops`.

Não por ordem burocrática. Por dependência real: B precisa de A para validar latents. C precisa de A para projetar. Se você codar B ou C sem A, vai reimplementar `exp_map_zero` inline de forma inconsistente e vai ter bugs matemáticos silenciosos que aparecem só em produção com dados reais.

A matemática primeiro. Tudo depende dela.
