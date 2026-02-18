O que muda com base no código real

### Whisper — eliminado do servidor

Confirmado. Whisper está em `EVA-Mobile-FZPN/android/.../whisper.h` — C++ on-device para STT offline no mobile. No servidor só existe Gemini Live API. A estratégia original estava errada.

### TextEncoder — já está 90% pronto

```
JÁ EXISTE na EVA-Mind:
  embeddings.go       → Gemini 3072D
  krylov_manager.go   → CompressVector(3072D → 64D)

FALTA só isso no NietzscheDB:
  exp_map_zero(vec_64d) → z_hyp ∈ R^64, ‖z‖ < 1.0

Esforço real: 1 função, ~15 linhas Rust.
TextEncoder é gratuito.
```

### AudioEncoder — a estratégia correta

A análise propôs 4 opções. Minha posição:

```
OPÇÃO C + D simultaneamente — não é OU, é E

z_audio ∈ R^128  ← mel spectrogram VAE (Rust, treinar)
                    preserva: prosódia, timbre, padrões emocionais
                    o "como foi dito"

z_text  ∈ R^64   ← Krylov existente + exp_map_zero (trivial)
                    preserva: semântica, significado
                    o "o que foi dito"

z_fused ∈ R^256  ← gyromidpoint(z_audio, z_text)
                    preserva: experiência completa da sessão de voz
```

Isso é biologicamente mais correto do que um encoder só. O cérebro processa prosódia e semântica em regiões separadas (córtex auditivo primário vs área de Wernicke) e funde depois. Você está replicando essa separação.

---

## Revisão da ordem de implementação

A ordem proposta no documento está correta. Só ajusto uma coisa:

```
PROPOSTO:         gRPC bridge na posição 4
MINHA POSIÇÃO:    gRPC bridge na posição 2, junto com SensoryMemory

Razão: você não consegue testar nada sem a bridge.
TextEncoder em Rust sem poder enviar dados da EVA-Mind é
código que você não consegue validar com dados reais.
```

Ordem final:

```
1. nietzsche-hyp-ops
     exp_map_zero, log_map_zero, mobius_add, gyromidpoint
     testes com valores conhecidos (tanh(1.0) = 0.7616...)

2. SensoryMemory struct + CF no RocksDB + gRPC bridge
     struct Node estendido
     CF "sensory" lazy-load
     rpc InsertSensory(node_id, modality, latent_bytes)
     rpc Reconstruct(node_id) → latent_bytes

3. TextEncoder
     EVA-Mind já faz: Gemini → Krylov → 64D
     NietzscheDB adiciona: exp_map_zero → z_hyp
     EVA-Mind envia 64D via gRPC, NietzscheDB projeta e salva

4. AudioEncoder (mel VAE em Rust)
     input: PCM 16kHz (já está no Redis da EVA)
     mel spectrogram (candle-transformers ou tch-rs)
     VAE 1D → z_audio ∈ R^128
     exp_map_zero → z_hyp
     ESTE é o gargalo — precisa de dados clínicos para treinar

5. FusionEncoder
     gyromidpoint(z_audio, z_text) → z_fused ∈ R^256
     só depois que 3 e 4 estão funcionando

6. LatentVector::degrade(energy)
     f32 → f16 → int8 → PQ → None

7. L-System tick estendido
     passo de degradação integrado

8. REM Consolidator estendido (Go + Rust)
     pipeline noturno adiciona snapshot/rollback de latents
     queue de latents → worker GPU para LoRA fine-tune assíncrono

9. ImageEncoder (VAE CNN)
     o único que precisa ser treinado do zero
     dados: imagens de exames clínicos reais
     deixa por último — maior esforço, menor urgência para EVA de voz

10. NQL extensions
      RECONSTRUCT statement
      SENSORY_DIST operator
```

---

## O gargalo real

O documento identifica corretamente: o AudioEncoder (mel VAE) é o gargalo. Não por complexidade de código — por dados.

Para treinar um VAE de áudio clínico que preserva prosódia de forma útil você precisa de áudios reais de sessões clínicas. O modelo treinado em LibriSpeech vai capturar fala geral, não vai capturar os padrões específicos que a EVA precisa detectar — tremor de Parkinson, pausa de depressão, variação de ansiedade.

Três opções para o problema de dados:

```
OPÇÃO 1 — Treinar com dados sintéticos primeiro
  Usar TTS (ElevenLabs, gTTS) para gerar áudios com variações
  de prosódia controladas. Valida a arquitetura antes de ter
  dados reais. Não produção, mas suficiente para Phase 11 MVP.

OPÇÃO 2 — Transfer learning de modelo pré-treinado
  EnCodec (Meta) ou SoundStream (Google) são VAEs de áudio
  com latents de qualidade. Frozen + Linear + exp_map_zero.
  Não captura prosódia clínica específica mas funciona day 1.

OPÇÃO 3 — Treinar com as sessões reais da EVA
  Cada sessão Twilio/browser já gera PCM 16kHz no Redis.
  Com consentimento LGPD adequado, acumular 1000+ sessões
  e treinar o VAE nos dados reais da aplicação.
  Melhor resultado, maior complexidade legal e de tempo.
```

Para MVP da Fase 11: Opção 2 (EnCodec frozen) desbloqueando tudo rapidamente. Opção 3 como evolução quando os dados estiverem disponíveis.

---

## O ponto mais importante do documento

Esta observação sobre o REM Consolidator:

```
rem_consolidator.go já faz:
  Hot memories → clustering → Krylov centroid → SemanticMemory → prune

Fase 11 estende com:
  snapshot latents → GPU queue → LoRA fine-tune → hot-reload
```

Isso significa que você não está construindo o ciclo de consolidação do zero. Está estendendo um pipeline noturno que já existe e já funciona. A estrutura de orquestração, o scheduling, o acesso ao Neo4j — tudo já está lá. A Fase 11 adiciona o tratamento de latents em cima de uma fundação sólida.

Isso reduz significativamente o risco de implementação. O pior cenário de uma fase nova é ter que construir toda a infraestrutura de suporte. Aqui você não precisa — só estende o que já existe.
