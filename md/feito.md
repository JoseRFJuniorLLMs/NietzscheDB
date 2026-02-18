## Principais Características

[](https://github.com/JoseRFJuniorLLMs/hyperspace-db#-key-features)


| 🧠**Contexto Infinito**               | Armazene milhões de vetores em milhares de coleções. Pague apenas pelo que usar.                                                                                                                               |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 💤**Núcleo sem servidor**            | **Remoção** automática por inatividade e **ativação instantânea** (inicialização a frio em ms).                                                                                                       |
| ⚡️**Desempenho Extremo**            | Construído com**Rust Nightly** e funções intrínsecas `std::simd` para obter o máximo desempenho em CPUs AVX2/Neon.                                                                                          |
| 🚀**Concorrência sem bloqueios**     | Novidade na versão 2.0: A arquitetura**ArcSwap** permite escalabilidade linear. Suporta **mais de 1000 clientes simultâneos** sem qualquer disputa por bloqueios.                                            |
| 📐**HNSW Hiperbólico Nativo**        | Uma implementação personalizada do algoritmo Hierarchical Navigable Small Worlds, matematicamente otimizada para a métrica de Poincaré (sem a sobrecarga dispendiosa do`acosh`).                              |
| 🔒**Seguro e Autenticado**            | Segurança nativa por chave de API (SHA-256) e controle de acesso baseado em funções para implantação em produção.                                                                                          |
| 🔎**Filtragem Avançada**             | Filtragem complexa de metadados com os operadores`Range` e `Match` usando Roaring Bitmaps.                                                                                                                        |
| 🤝**Cluster Federado**                | Replicação líder-seguidor com arquitetura**pronta para CRDT** para consistência distribuída e sincronização de borda.                                                                                     |
| 🧠**Busca Híbrida**                  | Combine a busca semântica (vetorial) com a busca por palavras-chave (lexical) usando a Fusão de Classificação Recíproca (RRF).                                                                               |
| 🏘️**Multilocação**                | Suporte nativo para separação lógica via**Collections** . Gerencie múltiplos índices vetoriais independentes em uma única instância.                                                                      |
| 🖥️**Painel Web**                    | Painel de controle integrado com visualização**da topologia do cluster** , métricas em tempo real e exploração de dados.                                                                                    |
| 📦**EscalarI8 e Binário**            | A quantização integrada**ScalarI8** e **Binária (1 bit)** reduz a ocupação de memória em até **64 vezes** , com velocidade impressionante.                                                            |
| ❄️**Armazenamento refrigerado**     | O carregamento lento (lazy loading) e**a remoção de itens ociosos (Idle Eviction)** garantem o uso mínimo de RAM, permitindo a escalabilidade para milhares de coleções em hardware com recursos limitados. |
| 🧵**Pipeline de Escrita Assíncrona** | A ingestão desacoplada com um WAL V2 garante a persistência de dados e metadados sem bloquear as leituras.                                                                                                      |
| 🛠️**Ajuste de tempo de execução** | Ajuste dinamicamente os parâmetros`ef_search` e `ef_construction` via gRPC sem reiniciar o servidor.                                                                                                             |
