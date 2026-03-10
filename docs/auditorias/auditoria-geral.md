# AUDITORIA GERAL — PROJETO EVA-Mind
**Data:** 2026-02-19
**Auditor:** Análise Automatizada Profunda
**Versão do Banco:** v43 (2026-02-13, 190 tabelas)

---

## ÍNDICE

1. [Visão Geral do Ecossistema](#1-visão-geral-do-ecossistema)
2. [Bugs Críticos por Projeto](#2-bugs-críticos-por-projeto)
3. [Código Não Usado / Morto](#3-código-não-usado--morto)
4. [O Que Pode Melhorar](#4-o-que-pode-melhorar)
5. [Banco de Dados — O Que Não é Usado](#5-banco-de-dados--o-que-não-é-usado)
6. [O Que Bloqueia a EVA](#6-o-que-bloqueia-a-eva)
7. [Análise das Memórias](#7-análise-das-memórias)
8. [Segurança — Alertas Críticos](#8-segurança--alertas-críticos)
9. [Score por Projeto](#9-score-por-projeto)

---

## 1. VISÃO GERAL DO ECOSSISTEMA

```
EVA-Mind (Go backend — WebSocket/gRPC)
  ├── EVA-Back (FastAPI Python — REST API, admin, pagamentos)
  ├── EVA-Front (React 18 — dashboard web)
  ├── EVA-Mobile (Flutter — app Android/iOS v1, legado)
  ├── EVA-Mobile-FZPN (Flutter — app Android/iOS v2, produção)
  ├── EVA-OS (Rust TUI — desktop/Redox OS)
  ├── EVA-Windows (Flutter desktop — Windows)
  ├── EVA-Kids (Angular 17 — plataforma educacional)
  ├── EVA-db (NietzscheDB 16 — 190 tabelas, pgvector)
  ├── IronMind (Flutter — inspeção industrial, fork do FZPN)
  └── Aurora-Platform (FastAPI + Next.js 15 — produto separado)

Infraestrutura de Memória:
  NietzscheDB (episodic_memories + pgvector)
  + NietzscheDB (grafo semântico de relações)
  + NietzscheDB (busca vetorial, embeddings)
  + NietzscheDB (cache FDPN em tempo real)

Deploy: GCP Cloud Run (backend REST) + VM GCP 136.113.25.218 (EVA-Mind)
        DigitalOcean 104.248.219.200 (fallback)
        Firebase Hosting (EVA-Front + EVA-Kids)
```

**Stack Principal:**
- Backend Core: Go (EVA-Mind) + Python FastAPI (EVA-Back)
- Frontend Web: React 18 + Vite + TailwindCSS
- Mobile: Flutter/Dart (Android + iOS)
- Desktop: Rust (EVA-OS) + Flutter Windows (EVA-Windows)
- Banco: NietzscheDB 16 + NietzscheDB + NietzscheDB + NietzscheDB

---

## 2. BUGS CRÍTICOS POR PROJETO

### 2.0 EVA-Mind (Go — Motor Principal)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| E01 | `_audit_db.go:17` + `main.go:88` | **`main()` duplicado em package main → projeto NÃO COMPILA** | CRÍTICO |
| E02 | `env_backup.txt` | Credenciais ao vivo commitadas: NietzscheDB, Twilio SID/Token, Google API Key | CRÍTICO |
| E03 | `_audit_db.go:15` | `const DB_URL` com IP de produção hardcoded no código | CRÍTICO |
| E04 | `_audit_db.go:44,72` | SQL injection via `fmt.Sprintf` com `funcName` sem escape | ALTO |
| E05 | `config.go:197` | `getEnvRequired("JWT_SECRET")` só loga warning se vazio → JWT secret vazio possível | CRÍTICO |
| E06 | `main.go` | `cfg.Validate()` definido mas nunca chamado → startup sem validação de env vars | ALTO |
| E07 | `internal/brainstem/auth` usa `jwt/v4`, `security/multitenancy` usa `jwt/v5` | Versões incompatíveis — tokens podem ser inválidos entre middlewares | ALTO |
| E08 | `internal/tools/handlers.go:604` | `handleGetAgendamentos` busca TODOS os pacientes e filtra em memória → data leak + N×O(n) | ALTO |
| E09 | `memory/orchestrator.go:91,95` | `// TODO: Store in NietzscheDB` e `// TODO: Store episodic memory` — operações core de memória não implementadas | ALTO |
| E10 | `hippocampus/memory/*.go` | ~8 funções NietzscheDB retornam zero values — `// TODO: Extraire corretamente do record` | ALTO |
| E11 | `cortex/explainability/pdf_generator.go:313` | Geração de PDF clínico é stub completo — `// TODO: Integrar com biblioteca` | ALTO |
| E12 | `pkg/safety/abuse_detector.go:216` | Todos os métodos de notificação são stubs vazios — abuso nunca reportado | ALTO |
| E13 | `cortex/lacan/unified_retrieval.go:365` | Debug mode ativado por nome hardcoded "José R F Junior" no código de produção | MÉDIO |
| E14 | `migrations/` | Numeração duplicada: 001, 002, 003, 016, 017, 018 cada um com 2 arquivos distintos | ALTO |
| E15 | `migrations/` | Lacunas: migrations 006, 028, 029, 031-034 ausentes | MÉDIO |
| E16 | `auth/handlers.go` | `Register`, `Me`, `RefreshToken` definidos mas NÃO registrados no router → endpoints mortos | MÉDIO |
| E17 | `video_websocket_handler.go:95` | `session.AttendantConn` acessado sem mutex → race condition em video calls | MÉDIO |
| E18 | `senses/signaling/websocket.go:1055` | `// HACK: Enviar para o próprio idoso (teste)` em path de produção | MÉDIO |
| E19 | `cortex/predictive/trajectory_engine.go:515` | Adherência a medicamento, horas de sono, isolamento: valores hardcoded (0.65, 5.5, 3) | MÉDIO |
| E20 | `docker-compose.infra.yml:12` | Senha NietzscheDB hardcoded `Debian23` no docker-compose commitado | ALTO |
| E21 | `config/` YAMLs | Arquivos YAML de config (`fdpn_boost.yaml`, `ram.yaml`, etc.) não são carregados pelo Go — são apenas documentação | INFO |
| E22 | 102+ | TODO/FIXME/HACK comments no codebase Go | — |

---

### 2.1 EVA-Back (FastAPI Python)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| B01 | `routes_auth.py:320` | `bcrypt` usado sem importação → **NameError em toda chamada de change_password** | CRÍTICO |
| B02 | `routes_auth.py:181` | `google_id` e `foto_url` não existem no model `Usuario` → Google OAuth quebrado | CRÍTICO |
| B03 | `webhook_tasks.py:87` | `transaction.amount_received` não existe no model → confirmação de pagamento falha | CRÍTICO |
| B04 | `webhook_tasks.py:257` | `Subscription.stripe_subscription_id` não existe → cancelamento Stripe nunca funciona | CRÍTICO |
| B05 | `webhook_tasks.py:123` | `user.subscription_tier` não existe no model `Usuario` → upgrade pós-pagamento falha | CRÍTICO |
| B06 | `routes_admin_payments.py:152` | `SELECT extend_subscription_period(...)` — função NietzscheDB não criada em nenhuma migration | CRÍTICO |
| B07 | `routes_voice.py` | Variável `text` (parâmetro) sobrescreve `from sqlalchemy import text` → TypeError em generate_speech | CRÍTICO |
| B08 | `reset_admin_password.py:110` | `//reste` — syntax error Python → arquivo não compila | ALTO |
| B09 | `webhook_tasks.py:264` | `sub.cancelled_at` não existe no model `Subscription` | ALTO |
| B10 | `models.py` | `NLPConversationAnalysis` tem FK para `conversation_sessions.id` e `conversation_messages.id` que não estão em nenhuma migration do EVA-Back | ALTO |
| B11 | `routes_checkout.py:64,117,163,219` | **Todos** os endpoints de checkout usam `user_id = 1` hardcoded — pagamentos de usuários reais nunca são registrados corretamente | CRÍTICO |
| B12 | `routes_admin_payments.py` | Todos os 4 endpoints admin não têm autenticação — qualquer pessoa pode aprovar/rejeitar transações financeiras | CRÍTICO |
| B13 | `routes_kids.py` | Race condition no saldo Satoshi: incremento não atômico sob concorrência | MÉDIO |
| B14 | `routes_kids_ws.py` | Estado WebSocket em memória (dict Python) — perdido em restart/deploy, incompatível com múltiplos workers | ALTO |
| B15 | `main.py:CORS` | `allow_origin_regex=".*"` + `allow_credentials=True` → qualquer site pode fazer requests autenticados | ALTO |
| B16 | `utils/security.py` | `SECRET_KEY` padrão hardcoded `"eva_secret_key_change_me_in_production"` → JWTs forjáveis se env var não setada | CRÍTICO |
| B17 | `Dockerfile` | EXPOSE 8080, `deploy.sh` usa 8000, `main.py` usa 8001 → inconsistência de porta em 3 lugares | MÉDIO |
| B18 | `routes_checkout.py:27` | `OpenNodePaymentService` importado duas vezes | BAIXO |
| B19 | `mental_health: C-SSRS ≥ 3` | Detecta risco de suicídio mas só loga — **protocolo de emergência não implementado** | CRÍTICO |

---

### 2.2 EVA-Front (React 18)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| F01 | `VoiceAssistant.jsx:8` | `import { GOOGLE_API_KEY } from '../configs/secrets.js'` — **arquivo não existe** → build quebra | CRÍTICO |
| F02 | `EvaContext.jsx:357` | `/historico/timiline/` (typo: "timiline") → Timeline sempre retorna 404 | ALTO |
| F03 | `cuidadoresService.js` + 5 outros | `localStorage.getItem('eva_token')` mas token salvo como `'token'` → **6 services sempre retornam 401** | CRÍTICO |
| F04 | `IncomingCallNotifier.jsx:128` | `connectWebSocket()` chamado duas vezes → 2 conexões WebSocket abertas, notificações duplicadas | ALTO |
| F05 | `App.jsx` | Rota `/call/:sessionId` fora do `ProtectedLayout` → qualquer pessoa acessa video calls sem auth | CRÍTICO |
| F06 | `FaceLoginModal.jsx` | Face login bem-sucedido para qualquer pessoa com câmera — nenhuma verificação real | CRÍTICO |
| F07 | `LoginPage.jsx:24` | `navigate('/dashboard')` sempre — loop de redirect para usuários `idoso` | MÉDIO |
| F08 | `ProfilePage.jsx` | `api.put('/auth/password')` — endpoint errado, deveria ser `PATCH /auth/change-password` | MÉDIO |
| F09 | `mentalHealthService.js` | `BASE_PATH = '/api/v1/mental-health'` + axios já tem `/api/v1` → URL dobrada → **todos os endpoints de saúde mental erram** | CRÍTICO |
| F10 | `clinicalDashboardService.js` | Mesmo problema de URL dobrada do F09 | CRÍTICO |
| F11 | `api.js` interceptor | Redirect 401 comentado → sessões expiradas falham silenciosamente | MÉDIO |
| F12 | `EvaContext.jsx` | `getInsightEva()` retorna `null` hardcoded — EVA insights nunca carregam | MÉDIO |
| F13 | `EvaContext.jsx` | `fetchAllIdosos` while(true) — loop infinito se backend retorna exatamente 200 items | ALTO |
| F14 | `VoiceAssistant.jsx` | AudioContexts nunca fechados em `stopConversation()` → memory leak progressivo | MÉDIO |
| F15 | `VideoCallPage.jsx` | Screen share: stream adquirido mas nunca substituído no peer connection → feature inoperante | MÉDIO |

---

### 2.3 EVA-Mobile-FZPN (Flutter — produção)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| M01 | `main.dart` | `health_sync_worker.dart` comentado, `workmanager` desabilitado → sincronização de saúde nunca ocorre | ALTO |
| M02 | `pubspec.yaml` | Modelos ONNX/Vosk comentados → scanner de medicamentos e voz offline não funcionam | ALTO |
| M03 | `sentinela_service.dart` | `_sendEmergencySMS` abre app de SMS ao invés de enviar silenciosamente — requer interação em emergência de queda | ALTO |
| M04 | `backend_selector.dart` | Usa HTTP (não HTTPS) apesar do `AppConfig` exigir HTTPS | MÉDIO |
| M05 | `sentinela_service.dart` | `triggerTestAlert` registra detecção 3 vezes → escala automaticamente para CRÍTICO em testes | BAIXO |

---

### 2.4 EVA-OS (Rust)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| O01 | `eva_mind.rs` | **CPF pessoal do criador hardcoded** (`64525430249`) como padrão de paciente | CRÍTICO |
| O02 | `timemachine/mod.rs` | `delete_today()` não deleta dados de hoje — apenas cleanup genérico | MÉDIO |
| O03 | `main.rs` | `frame_count` incrementado mas nunca usado | BAIXO |
| O04 | Fase 14/15 | STT offline (Vosk) e TTS local (piper-rs) não implementados | MÉDIO |

---

### 2.5 EVA-Windows (Flutter Desktop)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| W01 | `desktop_home.dart:19` | `ws://localhost:8080/v1/ws` hardcoded → nunca conecta em produção | CRÍTICO |
| W02 | `assets/` | `eva_avatar.riv` não existe → crash ao renderizar avatar | CRÍTICO |
| W03 | `audio_service_windows.dart` | `_calculateRMS()` retorna `0.5` constante → lip-sync do avatar não reage ao áudio | MÉDIO |

---

### 2.6 IronMind (Flutter Industrial)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| I01 | `yolo_engine.dart` | `_loadModel()` lança `UnsupportedError` sempre → **detecção YOLO completamente não funcional** | CRÍTICO |
| I02 | `yolo_engine.dart` | `_runInference()` retorna `[[[]]]` placeholder → todo processamento AI industrial é dead code | CRÍTICO |
| I03 | `android/cpp/` | `whisper.cpp` + `ggml.c` presentes mas sem `CMakeLists.txt` compilando → STT offline não compila | ALTO |
| I04 | `assets/` | Modelos ONNX ausentes (`yolo26n_ironmind_int8.onnx`, etc.) | CRÍTICO |

---

### 2.7 EVA-Kids (Angular 17)

| # | Arquivo | Descrição | Severidade |
|---|---------|-----------|-----------|
| K01 | `package.json` | `@google/generative-ai: ^0.1.1` — SDK desatualizado (versão atual: 0.24+), API mudou | ALTO |
| K02 | `ollama.component.ts` | Conecta a Ollama local → falha em produção sem proxy | MÉDIO |
| K03 | `app.routes.ts` | `voicegame3` usa `GameComponent` genérico com `maskedMode: true` — dead code/placeholder | BAIXO |
| K04 | `README.md` | Descreve outro projeto ("DashboardAnalyticsComponent") — nunca atualizado | INFO |

---

## 3. CÓDIGO NÃO USADO / MORTO

### EVA-Mind (Go)
| Item | Localização | Status |
|------|-------------|--------|
| `_audit_db.go` | Raiz do projeto — script de audit com `main()` duplicado | REMOVER IMEDIATAMENTE |
| `docs/legacy-python/api_server.py` + toda a pasta | FastAPI legado, credenciais hardcoded | Remover ou mover para fora do repo |
| `internal/cortex/personality/situation_modulator.go` | Marcado `// DEPRECATED: duplicado de cortex/situation/modulator.go` | Deletar |
| `sabedoria/scripts/seed_wisdom.go` | Duplicata de `cmd/seed_wisdom/main.go` | Deletar um dos dois |
| `internal/gemini/` (package) | Duplicata de `internal/cortex/gemini/`, usada só em testes com type mismatch | Consolidar ou deletar |
| `config/*.yaml` e `configs/core_memory.yaml` | Não são carregados pelo Go — são documentação disfarçada de config | Mover para `docs/` |
| `internal/brainstem/auth/handlers.go` | `Register`, `Me`, `RefreshToken` — não registrados no router | Registrar ou deletar |
| `MD/SRC/hebbian_updater.py`, `eva_routes.py` | Protótipos Python da fase de desenvolvimento | Deletar |
| Migrations 001/002/003/016/017/018 | Dois arquivos para cada número → comportamento indefinido no runner | Renumerar |

### EVA-Back
| Item | Localização | Status |
|------|-------------|--------|
| `database.py` (raiz) | SQLite models, nunca importado pela app | DEAD CODE — pode deletar |
| `routes_medication.py` | Comentado em `main.py` e `api/__init__.py` | DEAD CODE — pode deletar |
| `core/celery_app.py` | Versão antiga duplicada de `celery_app.py` | DEAD CODE — pode deletar |
| `routes_optional.py` | 7 endpoints retornam arrays vazios com `# TODO: implementar` | STUB — implementar ou deletar |
| `routes_placeholders.py` | 8 endpoints com dados fake hardcoded (ex: `"receita": 15000.0`) | STUB PERIGOSO — remove ou implementa |
| `eva-enterprise.tar.gz` | Binário no repositório git | REMOVE DO GIT |
| `tests/12.py`, `debug_seed.py`, etc. | Scripts ad-hoc de debug | Mover para `/scripts/dev/` ou deletar |
| `dashboard_atendente.html` | HTML standalone sem integração | Verificar uso ou deletar |

### EVA-Front
| Item | Localização | Status |
|------|-------------|--------|
| `src/ia/visual.ts`, `visual-3d.ts`, `sphere-shader.ts`, `utils.ts` | TypeScript com LitElement, sem tsconfig, nunca importados | DEAD CODE — pode deletar |
| `src/configs/evaFunctions.js` | Duplicata de `evaConfig.js`, nunca importado | DEAD CODE — pode deletar |
| `src/services/firebase.js` | Firebase inicializado mas `auth`, `db`, `storage` nunca usados | DEAD CODE — pode deletar |
| `src/pages/RelatoriosPage.jsx` | Página funcional sem rota | Adicionar rota ou deletar |
| `src/pages/MedicationManagementPage.jsx` | Página funcional sem rota | Adicionar rota ou deletar |
| `src/pages/ConversationAnalysisPage.jsx` | Página funcional sem rota | Adicionar rota ou deletar |
| `onEvaSpeech` prop em `VoiceAssistant` | Prop não passada em nenhum lugar | Dead prop — subtítulos nunca mostram |

### EVA-Mobile (v1)
| Item | Status |
|------|--------|
| Projeto inteiro | Supersedido pelo FZPN — considerar arquivar |
| `updateDeviceToken()` em api_service.dart | Duplicata de `syncTokenByCpf()`, marcado DEPRECATED |

### Aurora-Platform
| Item | Status |
|------|--------|
| `backend/api_gateway/__init__.py` | Vazio — API Gateway não implementado |
| `conversations/`, `personality_training/`, `visual_narratives/`, `video_avatar/` | Stubs vazios — funcionalidades no backlog |

---

## 4. O QUE PODE MELHORAR

### Arquitetura
1. **Unificar URLs de backend** — 4 IPs/domínios hardcoded em diferentes projetos (`136.113.25.218`, `104.248.219.200`, `eva-ia.org`, `localhost`). Centralizar em variáveis de ambiente com validação.
2. **Substituir deploys fragmentados** — 3 scripts de deploy (`build_eva_mind.sh`, `deploy_eva.sh`, `deploy_final.sh`) com portas conflitantes (8080/8090/8091). Criar 1 script único com Docker Compose.
3. **Alembic para migrações** — Alembic está instalado mas não usado. O banco está sendo gerenciado via SQL dumps manuais (v33 → v43). Migrar para Alembic auto-run em CI/CD.
4. **EVA-Back usar async driver consistente** — FastAPI usa `asyncpg` mas Celery usa `psycopg2`. Padronizar usando `psycopg3` (suporta sync e async).
5. **Remover EvaContext monolítico** — 500 linhas, 30+ states, 25+ funções num único Context. Dividir por domínio (auth, idosos, saúde, pagamentos).

### Segurança
6. **Rotacionar todas as credenciais expostas** — banco de dados, Firebase, Google API Key (ver seção 8).
7. **Implementar rate limiting** no EVA-Back (já existe tabela `rate_limits` no banco, mas não está integrada nas rotas).
8. **CORS restritivo** — trocar `allow_origin_regex=".*"` por lista de origens permitidas.
9. **Subscription tier do JWT** — tier lido do token (pode estar desatualizado). Verificar no banco a cada request crítico.

### Performance
10. **Cache de embeddings** — `embedding_cache.go` existe em EVA-Mind mas verificar se está configurado corretamente com NietzscheDB TTL.
11. **WebSocket signaling para Kubernetes** — mover de in-memory dict para NietzscheDB pub/sub para funcionar com múltiplos workers.
12. **Paginação em `fetchAllIdosos`** — o loop `while(true)` pode ser substituído por paginação cursor-based.

### Developer Experience
13. **Criar `.env.example`** em todos os projetos — atualmente só código-fonte documenta as vars necessárias.
14. **Separar `requirements.txt` dev/prod** — `faker`, `pytest`, `aiosqlite` estão em prod.
15. **Adicionar `tsconfig.json`** ao EVA-Front ou remover os arquivos `.ts` que não compilam.
16. **Padronizar portas** — definir uma vez (ex: `docker-compose.yml`) e referenciar em todos os outros lugares.

### Qualidade de Código
17. **Implementar error boundaries** no React — erros em componentes profundos crasham a UI inteira.
18. **Adicionar retry/circuit breaker no frontend** — hoje uma falha de API resulta em tela branca silenciosa.
19. **EVA-Kids atualizar Gemini SDK** — de `^0.1.1` para `^0.24+` (API incompatível).
20. **Documentação de API** — `doc/endpoint.txt` no EVA-Front tem ~250 endpoints mas não está sincronizado com o backend real.

---

## 5. BANCO DE DADOS — O QUE NÃO É USADO

### Tabelas Duplicadas / Redundantes (v43 — 190 tabelas)

| Grupo | Tabelas Duplicadas | Recomendação |
|-------|-------------------|--------------|
| Ferramentas/Funções AI | `funcoes`, `function_definitions`, `available_tools`, `tool_definitions` | Consolidar em 1 tabela |
| Configuração | `configuracoes_sistema`, `system_config` | Manter só `system_config`, migrar dados |
| Sinais Vitais | `sinais_vitais`, `sinais_vitais_health` | Mesclar com flag `source` (manual/healthkit) |
| Assinaturas | `assinaturas`, `subscriptions` | `assinaturas` é legado — migrar e dropar |
| Prompts | `prompts`, `prompt_templates`, `prompt_templates_personalized`, `system_prompts` | Consolidar com campo `type` |
| Auditoria | `audit_logs`, `auth_audit_logs`, `lgpd_audit_log` | Manter separados mas documentar escopo |
| Interações drogas | `interacoes_risco`, `medication_side_effects` | Verificar se são realmente distintos |

### Tabelas Potencialmente Órfãs

| Tabela | Motivo de Suspeita |
|--------|-------------------|
| `historico` | Estrutura mínima, possível placeholder legacy |
| `project_codebase` | Armazena código-fonte dentro do banco — não é pattern recomendado |
| `project_metadata` | Similar ao acima |
| `rate_limits` | Tabela existe mas não está sendo consultada por nenhuma rota ativa |
| `funcoes` | Versão legacy de `function_definitions` |
| `pagamentos` | Possível legado de `transactions` |

### Crescimento Acelerado (Risco)

O banco **quase dobrou de tamanho** em 3 semanas:
- 2026-01-24: **104 tabelas** (v33)
- 2026-02-13: **190 tabelas** (v43)

Principais adições: framework Lacaniano (10+ tabelas), sistema de personas (5), enneagrama (4), pesquisa clínica (4), ML avançado (7), legado digital (6), LGPD (6), gestão de API (3), ferramentas AI (4).

**Recomendação:** Definir critério de "tabela entra em produção" — migração formal + código que a usa + testes.

### Ausência de Alembic Operacional

11 dumps SQL (v33–v43) mas apenas **1 arquivo de migration formal** (`medication_visual_system.sql`). O versionamento do schema está sendo feito por dumps completos, não por migrações incrementais. Isso impossibilita:
- Rollback granular
- Deploy automatizado do schema
- Rastreamento de quem alterou o quê

---

## 6. O QUE BLOQUEIA A EVA

### Bloqueadores IMEDIATOS (sistema não funciona sem resolver)

| # | Componente | Bloqueador | Impacto |
|---|-----------|-----------|---------|
| 🔴 0 | EVA-Mind | `_audit_db.go` tem `main()` duplicado → **Go NÃO COMPILA** | Motor principal inutilizável |
| 🔴 1 | EVA-Front | `secrets.js` não existe → build quebra | VoiceAssistant inutilizável |
| 🔴 2 | EVA-Front | 6 services usam `eva_token` (key errada) → 401 em tudo | Cuidadores, medicamentos, protocolos, relatórios, pagamentos, vozes — todos quebrados |
| 🔴 3 | EVA-Front | URL dobrada em mental health service | Saúde mental e dashboard clínico inacessíveis |
| 🔴 4 | EVA-Back | `bcrypt` NameError → change_password crashe | Usuários não conseguem trocar senha |
| 🔴 5 | EVA-Back | Colunas ausentes em `Usuario` (`google_id`, `foto_url`, `subscription_tier`) | Login Google e upgrade de plano quebrados |
| 🔴 6 | EVA-Back | Todos checkouts com `user_id = 1` | Pagamentos não atribuídos a usuários reais |
| 🔴 7 | EVA-Back | Admin endpoints sem autenticação | Risco financeiro imediato |
| 🔴 8 | EVA-Back | `extend_subscription_period` SQL function não existe | Aprovação de pagamentos manual crasha |
| 🔴 9 | EVA-Back | Stripe/Asaas webhook tasks referenciam colunas inexistentes | Pagamentos nunca confirmados automaticamente |
| 🔴 10 | EVA-Windows | Arquivo `eva_avatar.riv` ausente | App crasha ao iniciar |
| 🔴 11 | EVA-Windows | WebSocket hardcoded `localhost:8080` | Nunca conecta em produção |
| 🔴 12 | IronMind | YOLO inference é stub → retorna `[[[]]]` | IA industrial completamente non-funcional |

### Bloqueadores de PRODUÇÃO (funciona localmente, falha em prod)

| # | Componente | Bloqueador |
|---|-----------|-----------|
| 🟠 13 | EVA-Mind/todos os clients | CPF hardcoded `64525430249` em EVA-OS |
| 🟠 14 | EVA-Back | Celery sem process management — pagamentos assíncronos não processam |
| 🟠 15 | EVA-Mobile-FZPN | Health sync worker desabilitado |
| 🟠 16 | EVA-Back | PORT inconsistente (8080/8001/8000) no Dockerfile vs deploy |
| 🟠 17 | EVA-Back | WebSocket signaling em memória — incompatível com múltiplos workers Cloud Run |
| 🟠 18 | EVA-Kids | Gemini SDK v0.1.1 — API incompatível com versões atuais |

### Bloqueadores de SEGURANÇA (devem ser resolvidos urgente)

| # | Componente | Bloqueador |
|---|-----------|-----------|
| 🔐 19 | EVA-Back | `env_backup.txt` com credenciais DB ao vivo commitado no git |
| 🔐 20 | EVA-Front | `firebase.js` com API keys hardcoded no código-fonte |
| 🔐 21 | EVA-Front | `.env` com `VITE_GOOGLE_API_KEY` possivelmente no git |
| 🔐 22 | EVA-OS | CPF pessoal do criador commitado no código-fonte |
| 🔐 23 | Aurora | `auth_login.txt` + `.env` possivelmente com credenciais no repo |
| 🔐 24 | EVA-Back | SECRET_KEY JWT com valor padrão público |
| 🔐 25 | EVA-Front | FaceLogin aceita qualquer câmera → bypass de autenticação |
| 🔐 26 | EVA-Front | Video call route sem autenticação |

---

## 7. ANÁLISE DAS MEMÓRIAS

### Arquitetura de Memória (EVA-Mind — Go)

A EVA implementa um sistema de memória biologicamente inspirado e altamente sofisticado:

```
Voz → FDPN Engine → Krylov Compression → Spectral Clustering → REM Consolidation
                                                                      ↓
                                              NietzscheDB ← Grafo semântico de relações
                                              NietzscheDB ← Embeddings vetoriais (1536D → 64D)
                                              NietzscheDB ← Memórias episódicas
                                              NietzscheDB ← Cache de ativação em tempo real
```

### Componentes de Memória

| Componente | Arquivo | Status |
|-----------|---------|--------|
| **FDPN Engine** | `hippocampus/memory/fdpn_engine.go` | Implementado — NietzscheDB + NietzscheDB + NietzscheDB, 10 threads, threshold 0.3 |
| **Krylov Compression** | `memory/krylov/krylov_manager.go` | Implementado — comprime 1536D → 64D |
| **REM Consolidation** | `memory/consolidation/rem_consolidator.go` | Implementado — episódico → clustering espectral → semântico NietzscheDB |
| **Hebbian Learning** | `memory/consolidation/hebbian.go` | Implementado |
| **Memory Pruning** | `memory/consolidation/pruning.go` | Implementado |
| **Spaced Repetition** | `hippocampus/spaced/spaced_repetition.go` | Implementado |
| **Zettelkasten** | `hippocampus/zettelkasten/zettel_service.go` | Implementado |
| **Persistent Homology** | `hippocampus/topology/persistent_homology.go` | Implementado (TDA — análise topológica) |

### Subsistema "Superhuman" (Psicanalítico)

| Componente | Arquivo | Descrição |
|-----------|---------|-----------|
| **Lacanian Mirror** | `superhuman/lacanian_mirror.go` | Simulação do estágio do espelho lacaniano |
| **Enneagram Service** | `superhuman/enneagram_service.go` | Processamento de tipos enneagrama |
| **Narrative Weaver** | `superhuman/narrative_weaver.go` | Construção de narrativa a partir de memórias |
| **Self Core** | `superhuman/self_core_service.go` | Identidade/conceito-de-si da EVA |
| **Consciousness** | `superhuman/consciousness_service.go` | Simulação de consciência |

### Base de Conhecimento (NietzscheDB Collections)

A EVA busca sabiamente em coleções de sabedoria:
- Histórias de Nasrudin
- Fábulas de Esopo
- Ensinamentos de Gurdjieff
- Insights de Osho
- Koans Zen
- Aforismos de Nietzsche
- Meditações Estoicas
- Poemas de Rumi
- Scripts de respiração
- Scripts de hipnose

### Banco de Dados de Memórias (v43)

| Tabela | Descrição |
|--------|-----------|
| `episodic_memories` | Memórias episódicas com pgvector embeddings |
| `idosos_memoria` | Perfil de memória por paciente |
| `conversation_messages` | Mensagens individuais de conversas |
| `conversation_sessions` | Sessões de conversa |
| `conversation_summaries` | Resumos AI das conversas |
| `analise_gemini` | Resultados de análise Gemini |
| `analise_audio_avancada` | Análise avançada de áudio |
| `nlp_conversation_analysis` | Análise NLP de conversas |
| `session_syntheses` | Sínteses de sessão (NOVO v43) |
| `personality_snapshots` | Evolução de personalidade (NOVO v43) |
| `significantes_recorrentes` | Significantes recorrentes (psicanalítico) |
| `transferencia_markers` | Marcadores de transferência |

### Análise da Arquitetura de Memória

**Pontos fortes:**
- Sistema multi-store altamente sofisticado (NietzscheDB + NietzscheDB + NietzscheDB + NietzscheDB)
- Inspiração biológica real (REM, Hebbian, FDPN)
- Compressão Krylov economiza espaço de embedding
- Análise topológica (homologia persistente) é estado da arte

**Pontos de atenção:**
- Framework Lacaniano no banco: 10+ tabelas adicionadas em v41 — verificar se EVA-Mind as usa ativamente
- `project_codebase` e `project_metadata` armazenam código-fonte dentro do banco — remover
- Tokens OAuth2 (`google_refresh_token`, `google_access_token`) na tabela `idosos` — mover para secrets storage
- `external_system_credentials` no banco — verificar se está criptografado em repouso

---

## 8. SEGURANÇA — ALERTAS CRÍTICOS

### 🔴 AÇÃO IMEDIATA NECESSÁRIA

```
1. ROTACIONAR AGORA:
   - Credenciais NietzscheDB em env_backup.txt (IP: 34.39.249.108)
   - Google API Key: AIzaSyBlem2g_EFVLTt3Fb1AofF1EOAf05YPo3U (.env EVA-Front)
   - Firebase API Key: AIzaSyAaEeKNGxz_1FOCT4SmP2CDIKx4zLLCDC8 (firebase.js hardcoded)

2. REVOGAR E REGENERAR:
   - CPF pessoal em eva_mind.rs — trocar por var de ambiente
   - SECRET_KEY JWT — forçar via env var sem fallback hardcoded

3. AUDITORIA:
   - Aurora-Platform: auth_login.txt e .env (verificar conteúdo e remover do git)
   - Tokens pg_restore (\restrict) nos SQL dumps commitados no git
```

### Credenciais Expostas Confirmadas

| Local | Tipo | Risco |
|-------|------|-------|
| `EVA-Mind/env_backup.txt` | NietzscheDB + **Twilio SID/Token** + Google API Key ao vivo | CRÍTICO |
| `EVA-Mind/_audit_db.go:15` | `const DB_URL` com IP de produção (104.248.x.x) hardcoded | CRÍTICO |
| `EVA-Mind/docker-compose.infra.yml` | NietzscheDB password `Debian23` hardcoded | ALTO |
| `EVA-Back/eva-enterprise/env_backup.txt` | Credenciais DB NietzscheDB ao vivo | CRÍTICO |
| `EVA-Back/deploy.sh` | Credenciais DB hardcoded no script | CRÍTICO |
| `EVA-Front/src/services/firebase.js` | Firebase API Key no código-fonte | ALTO |
| `EVA-Front/.env` | Google Gemini API Key | ALTO (se no git) |
| `EVA-OS/eva-daemon/src/eva_mind.rs` | CPF pessoal do criador | MÉDIO |
| `EVA-db/v33.sql` até `v43.sql` | Tokens `\restrict` pg_restore | MÉDIO |

---

## 9. SCORE POR PROJETO

| Projeto | Funcional? | Completo? | Bugs Críticos | Segurança | Score Geral |
|---------|-----------|-----------|---------------|-----------|-------------|
| **EVA-Mind** (Go) | ❌ Não compila | 80% | **22** | 🔴 Crítico | **D+** |
| **EVA-Back** (FastAPI) | ⚠️ Parcial | 70% | **13** | 🔴 Crítico | **D** |
| **EVA-Front** (React) | ⚠️ Parcial | 65% | **15** | 🟠 Alto | **D+** |
| **EVA-Mobile-FZPN** | ✅ Core OK | 80% | 5 | 🟡 Médio | **B** |
| **EVA-Mobile** (v1) | ✅ Core OK | 60% | 2 | 🟡 Médio | **C** (legado) |
| **EVA-OS** | ✅ Demo OK | 70% | 1 crítico | 🔴 CPF exposto | **C+** |
| **EVA-Windows** | ❌ Não | 40% | 2 críticos | 🟡 Médio | **F** |
| **EVA-Kids** | ✅ Parcial | 70% | 1 alto | 🟡 Médio | **C+** |
| **IronMind** | ❌ AI stub | 50% | 2 críticos | 🟡 Médio | **D+** |
| **Aurora-Platform** | ⚠️ MVP | 35% | 0 | 🔴 .env exposto | **D** |
| **EVA-db** | ✅ Operacional | 90% | 0 | 🟠 Credenciais | **B+** |

---

## RESUMO EXECUTIVO

### O que funciona bem
- EVA-Mind (Go): motor de memória e WebSocket são sofisticados e bem construídos
- EVA-db: estrutura robusta com 190 tabelas cobrindo todos os domínios
- EVA-Mobile-FZPN: detecção de quedas (Sentinela) e core de chamadas estável
- Arquitetura de memória multi-store (NietzscheDB + NietzscheDB + NietzscheDB + NietzscheDB) é estado da arte

### Prioridade de Correção

**HOJE (EVA-Mind não compila):**
0. Deletar `_audit_db.go` do EVA-Mind — causa `main()` duplicado e impede qualquer build Go
0b. Remover `env_backup.txt` do EVA-Mind e do git history (BFG Repo-Cleaner)
0c. Rotacionar credenciais Twilio, NietzscheDB (ambos IPs), Google API Key

**SEMANA 1 (Bloqueadores de funcionamento):**
1. Criar `secrets.js` no EVA-Front (ou refatorar para `import.meta.env`)
2. Corrigir chave localStorage `eva_token` → `token` nos 6 services
3. Corrigir URL dobrada em `mentalHealthService.js` e `clinicalDashboardService.js`
4. Corrigir import `bcrypt` em `routes_auth.py`
5. Adicionar colunas `google_id`, `foto_url`, `subscription_tier` em migration
6. Corrigir `user_id` nos endpoints de checkout
7. Adicionar autenticação nos endpoints admin de pagamento
8. Criar função SQL `extend_subscription_period`
9. Rotacionar todas as credenciais expostas

**SEMANA 2 (Qualidade e segurança):**
10. Corrigir portas inconsistentes (Dockerfile, deploy.sh, main.py)
11. Implementar Alembic migrations em substituição a dumps SQL
12. Adicionar `eva_avatar.riv` ao EVA-Windows
13. Corrigir WebSocket URL hardcoded em EVA-Windows
14. Remover `env_backup.txt` do repositório
15. Implementar protocolo de emergência para detecção de risco suicida

**SEMANA 3+ (Melhorias e completude):**
16. Consolidar tabelas redundantes no banco
17. Implementar Celery process management (docker-compose com worker)
18. Implementar YOLO inference no IronMind (modelos ONNX + bridge)
19. Unificar scripts de deploy
20. Atualizar Gemini SDK no EVA-Kids

---

*Auditoria gerada em 2026-02-19 por análise estática automatizada de código-fonte.*
*Próxima auditoria recomendada: após resolução dos bloqueadores críticos (estimativa: 2 semanas).*
