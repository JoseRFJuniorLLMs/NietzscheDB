//! seed_100 — insere 100 nós filosóficos + ~150 edges no NietzscheDB.
//!
//! Estrutura do grafo (5 clusters):
//!   A. Pré-Socráticos   (20 nós) — coordenadas próximas de (0.1, 0.1, 0.1, 0.1)
//!   B. Idealistas       (20 nós) — coordenadas próximas de (0.5, 0.1, 0.1, 0.1)
//!   C. Existencialistas (20 nós) — coordenadas próximas de (0.1, 0.5, 0.1, 0.1)
//!   D. Analíticos       (20 nós) — coordenadas próximas de (0.1, 0.1, 0.5, 0.1)
//!   E. Filósofos Nietzsche-core (20 nós) — centro da bola de Poincaré (~0)
//!
//! Edges:
//!   • Intra-cluster: todos os nós de um cluster conectados em anel
//!   • Inter-cluster: um "hub" por cluster liga aos 4 outros hubs
//!   • Hierárquicos: nós de energia alta → nós de energia baixa do mesmo cluster
//!
//! Uso:
//!   # Servidor rodando em localhost:50051
//!   cargo run -p nietzsche-sdk --example seed_100
//!
//!   # Servidor em outro host
//!   NIETZSCHE_ADDR=http://10.0.1.5:50051 cargo run -p nietzsche-sdk --example seed_100

use nietzsche_sdk::{InsertEdgeParams, InsertNodeParams, NietzscheClient};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde_json::json;
use uuid::Uuid;

// ── Dados dos filósofos ───────────────────────────────────────────────────────

struct Philosopher {
    name:      &'static str,
    school:    &'static str,
    period:    &'static str,
    idea:      &'static str,
    node_type: &'static str,
    energy:    f32,
}

const PHILOSOPHERS: &[Philosopher] = &[
    // ── A. Pré-Socráticos ─────────────────────────────────────────────────
    Philosopher { name: "Tales de Mileto",   school: "Pre-Socratic", period: "624-546 aC", idea: "A água é o princípio de todas as coisas",           node_type: "Concept",  energy: 0.85 },
    Philosopher { name: "Anaximandro",       school: "Pre-Socratic", period: "610-546 aC", idea: "O apeiron: o ilimitado é a origem de tudo",          node_type: "Concept",  energy: 0.80 },
    Philosopher { name: "Anaxímenes",        school: "Pre-Socratic", period: "585-525 aC", idea: "O ar como substância primordial",                   node_type: "Concept",  energy: 0.75 },
    Philosopher { name: "Heráclito",         school: "Pre-Socratic", period: "535-475 aC", idea: "Tudo flui; o fogo é a essência da mudança",         node_type: "Concept",  energy: 0.95 },
    Philosopher { name: "Parmênides",        school: "Pre-Socratic", period: "515-450 aC", idea: "O ser é uno e imutável",                            node_type: "Concept",  energy: 0.90 },
    Philosopher { name: "Zenão de Eleia",    school: "Pre-Socratic", period: "490-430 aC", idea: "Os paradoxos do movimento e do infinito",           node_type: "Concept",  energy: 0.88 },
    Philosopher { name: "Empédocles",        school: "Pre-Socratic", period: "494-434 aC", idea: "Quatro raízes: terra, água, fogo e ar",             node_type: "Concept",  energy: 0.82 },
    Philosopher { name: "Anaxágoras",        school: "Pre-Socratic", period: "500-428 aC", idea: "O Nous como inteligência ordenadora",               node_type: "Concept",  energy: 0.78 },
    Philosopher { name: "Demócrito",         school: "Pre-Socratic", period: "460-370 aC", idea: "Tudo é composto de átomos indivisíveis",            node_type: "Concept",  energy: 0.92 },
    Philosopher { name: "Leucipo",           school: "Pre-Socratic", period: "490-430 aC", idea: "Fundador do atomismo antigo",                       node_type: "Concept",  energy: 0.70 },
    Philosopher { name: "Pitágoras",         school: "Pre-Socratic", period: "570-495 aC", idea: "Os números são a realidade de todas as coisas",     node_type: "Concept",  energy: 0.93 },
    Philosopher { name: "Filolau",           school: "Pre-Socratic", period: "470-385 aC", idea: "O cosmos é harmonicamente numérico",                node_type: "Concept",  energy: 0.71 },
    Philosopher { name: "Protágoras",        school: "Pre-Socratic", period: "490-420 aC", idea: "O homem é a medida de todas as coisas",             node_type: "Semantic", energy: 0.89 },
    Philosopher { name: "Górgias",           school: "Pre-Socratic", period: "483-375 aC", idea: "Nada existe; se existisse não poderíamos conhecer", node_type: "Semantic", energy: 0.76 },
    Philosopher { name: "Sócrates",          school: "Pre-Socratic", period: "470-399 aC", idea: "Conhece-te a ti mesmo",                            node_type: "Semantic", energy: 0.99 },
    Philosopher { name: "Diógenes de Sínope",school: "Pre-Socratic", period: "412-323 aC", idea: "Virtude como razão e autossuficiência",             node_type: "Semantic", energy: 0.83 },
    Philosopher { name: "Antístenes",        school: "Pre-Socratic", period: "445-365 aC", idea: "A virtude pode ser ensinada",                      node_type: "Semantic", energy: 0.72 },
    Philosopher { name: "Xenofonte",         school: "Pre-Socratic", period: "430-354 aC", idea: "Memórias socráticas e filosofia prática",           node_type: "Episodic", energy: 0.68 },
    Philosopher { name: "Crátilo",           school: "Pre-Socratic", period: "440-370 aC", idea: "Radicalização do fluxo heraclítico",                node_type: "Episodic", energy: 0.65 },
    Philosopher { name: "Melisso de Samos",  school: "Pre-Socratic", period: "470-430 aC", idea: "O ser é infinito e eterno",                        node_type: "Concept",  energy: 0.66 },

    // ── B. Idealistas ─────────────────────────────────────────────────────
    Philosopher { name: "Platão",            school: "Idealist", period: "428-348 aC", idea: "O mundo das Formas é a verdadeira realidade",     node_type: "Concept",  energy: 0.99 },
    Philosopher { name: "Aristóteles",       school: "Idealist", period: "384-322 aC", idea: "A forma está na matéria; ser como ser",           node_type: "Concept",  energy: 0.98 },
    Philosopher { name: "Plotino",           school: "Idealist", period: "204-270",    idea: "O Uno é a fonte de toda existência",              node_type: "Concept",  energy: 0.91 },
    Philosopher { name: "Agostinho",         school: "Idealist", period: "354-430",    idea: "A realidade como participação divina",            node_type: "Concept",  energy: 0.87 },
    Philosopher { name: "Tomás de Aquino",   school: "Idealist", period: "1225-1274",  idea: "Síntese aristotélica com a fé cristã",           node_type: "Concept",  energy: 0.86 },
    Philosopher { name: "Descartes",         school: "Idealist", period: "1596-1650",  idea: "Cogito ergo sum — dualismo mente-corpo",         node_type: "Concept",  energy: 0.95 },
    Philosopher { name: "Spinoza",           school: "Idealist", period: "1632-1677",  idea: "Deus sive Natura — substância única infinita",   node_type: "Concept",  energy: 0.93 },
    Philosopher { name: "Leibniz",           school: "Idealist", period: "1646-1716",  idea: "Mônadas e harmonia pré-estabelecida",            node_type: "Concept",  energy: 0.92 },
    Philosopher { name: "Locke",             school: "Idealist", period: "1632-1704",  idea: "A mente como tábula rasa; empirismo",            node_type: "Semantic", energy: 0.88 },
    Philosopher { name: "Berkeley",          school: "Idealist", period: "1685-1753",  idea: "Esse est percipi — existir é ser percebido",     node_type: "Semantic", energy: 0.84 },
    Philosopher { name: "Hume",              school: "Idealist", period: "1711-1776",  idea: "Ceticismo e teoria das impressões",              node_type: "Semantic", energy: 0.90 },
    Philosopher { name: "Kant",              school: "Idealist", period: "1724-1804",  idea: "Imperativo categórico e estruturas do conhecimento", node_type: "Concept", energy: 0.97 },
    Philosopher { name: "Fichte",            school: "Idealist", period: "1762-1814",  idea: "O Eu absoluto como fundamento da realidade",     node_type: "Concept",  energy: 0.81 },
    Philosopher { name: "Schelling",         school: "Idealist", period: "1775-1854",  idea: "Identidade de natureza e espírito",              node_type: "Concept",  energy: 0.79 },
    Philosopher { name: "Hegel",             school: "Idealist", period: "1770-1831",  idea: "Dialética: tese, antítese, síntese",             node_type: "Concept",  energy: 0.96 },
    Philosopher { name: "Schopenhauer",      school: "Idealist", period: "1788-1860",  idea: "O mundo como vontade e representação",           node_type: "Concept",  energy: 0.94 },
    Philosopher { name: "Marx",              school: "Idealist", period: "1818-1883",  idea: "Materialismo histórico e dialético",             node_type: "Semantic", energy: 0.92 },
    Philosopher { name: "Engels",            school: "Idealist", period: "1820-1895",  idea: "Síntese marxista com dialética hegeliana",       node_type: "Semantic", energy: 0.75 },
    Philosopher { name: "Kierkegaard",       school: "Idealist", period: "1813-1855",  idea: "A subjetividade é a verdade; estágios da existência", node_type: "Episodic", energy: 0.89 },
    Philosopher { name: "Feuerbach",         school: "Idealist", period: "1804-1872",  idea: "A religião é projeção da essência humana",       node_type: "Semantic", energy: 0.73 },

    // ── C. Existencialistas ───────────────────────────────────────────────
    Philosopher { name: "Nietzsche",         school: "Existentialist", period: "1844-1900", idea: "Vontade de poder; Übermensch; eterno retorno",  node_type: "Concept",  energy: 1.00 },
    Philosopher { name: "Husserl",           school: "Existentialist", period: "1859-1938", idea: "Fenomenologia: às coisas mesmas",              node_type: "Concept",  energy: 0.91 },
    Philosopher { name: "Heidegger",         school: "Existentialist", period: "1889-1976", idea: "Dasein — o ser-no-mundo e o cuidado",         node_type: "Concept",  energy: 0.96 },
    Philosopher { name: "Sartre",            school: "Existentialist", period: "1905-1980", idea: "A existência precede a essência",              node_type: "Concept",  energy: 0.95 },
    Philosopher { name: "Camus",             school: "Existentialist", period: "1913-1960", idea: "O absurdo e a revolta criativa",               node_type: "Concept",  energy: 0.93 },
    Philosopher { name: "Simone de Beauvoir",school: "Existentialist", period: "1908-1986", idea: "O segundo sexo; ética da ambiguidade",         node_type: "Semantic", energy: 0.92 },
    Philosopher { name: "Merleau-Ponty",     school: "Existentialist", period: "1908-1961", idea: "Fenomenologia da percepção e corporeidade",    node_type: "Semantic", energy: 0.87 },
    Philosopher { name: "Gadamer",           school: "Existentialist", period: "1900-2002", idea: "Hermenêutica e fusão de horizontes",          node_type: "Semantic", energy: 0.82 },
    Philosopher { name: "Ricoeur",           school: "Existentialist", period: "1913-2005", idea: "Narratividade e identidade pessoal",          node_type: "Semantic", energy: 0.80 },
    Philosopher { name: "Levinas",           school: "Existentialist", period: "1906-1995", idea: "Ética como filosofia primeira; o rosto do Outro", node_type: "Concept", energy: 0.88 },
    Philosopher { name: "Jaspers",           school: "Existentialist", period: "1883-1969", idea: "Situações-limite e existência autêntica",      node_type: "Concept",  energy: 0.83 },
    Philosopher { name: "Buber",             school: "Existentialist", period: "1878-1965", idea: "Relação Eu-Tu e diálogo autêntico",            node_type: "Semantic", energy: 0.79 },
    Philosopher { name: "Arendt",            school: "Existentialist", period: "1906-1975", idea: "A condição humana; poder como ação coletiva",  node_type: "Semantic", energy: 0.91 },
    Philosopher { name: "Derrida",           school: "Existentialist", period: "1930-2004", idea: "Desconstrução e différance",                  node_type: "Semantic", energy: 0.85 },
    Philosopher { name: "Foucault",          school: "Existentialist", period: "1926-1984", idea: "Arqueologia do saber e genealogia do poder",   node_type: "Semantic", energy: 0.90 },
    Philosopher { name: "Deleuze",           school: "Existentialist", period: "1925-1995", idea: "Rizoma; devir; diferença e repetição",         node_type: "Concept",  energy: 0.88 },
    Philosopher { name: "Baudrillard",       school: "Existentialist", period: "1929-2007", idea: "Simulacros e hiperrealidade",                  node_type: "Semantic", energy: 0.78 },
    Philosopher { name: "Lyotard",           school: "Existentialist", period: "1924-1998", idea: "O pós-moderno; metanarativas em colapso",      node_type: "Semantic", energy: 0.74 },
    Philosopher { name: "Žižek",             school: "Existentialist", period: "1949-hoje", idea: "Ideologia lacaniana e análise do capital",     node_type: "Semantic", energy: 0.76 },
    Philosopher { name: "Badiou",            school: "Existentialist", period: "1937-hoje", idea: "Ser e evento; ontologia matemática",           node_type: "Concept",  energy: 0.81 },

    // ── D. Analíticos ─────────────────────────────────────────────────────
    Philosopher { name: "Frege",             school: "Analytic", period: "1848-1925", idea: "Lógica de predicados; semântica do sentido",     node_type: "Concept",  energy: 0.93 },
    Philosopher { name: "Russell",           school: "Analytic", period: "1872-1970", idea: "Atomismo lógico; teoria dos tipos",              node_type: "Concept",  energy: 0.95 },
    Philosopher { name: "Wittgenstein",      school: "Analytic", period: "1889-1951", idea: "Jogos de linguagem; os limites da linguagem",    node_type: "Concept",  energy: 0.97 },
    Philosopher { name: "Moore",             school: "Analytic", period: "1873-1958", idea: "Refutação do idealismo; senso comum",            node_type: "Concept",  energy: 0.84 },
    Philosopher { name: "Carnap",            school: "Analytic", period: "1891-1970", idea: "Empirismo lógico e verificacionismo",            node_type: "Concept",  energy: 0.86 },
    Philosopher { name: "Popper",            school: "Analytic", period: "1902-1994", idea: "Falsificacionismo e sociedade aberta",           node_type: "Semantic", energy: 0.92 },
    Philosopher { name: "Quine",             school: "Analytic", period: "1908-2000", idea: "Dois dogmas do empirismo; holismo semântico",    node_type: "Semantic", energy: 0.89 },
    Philosopher { name: "Austin",            school: "Analytic", period: "1911-1960", idea: "Atos de fala e filosofia da linguagem ordinária", node_type: "Semantic", energy: 0.80 },
    Philosopher { name: "Kripke",            school: "Analytic", period: "1940-hoje", idea: "Nomeação e necessidade; mundos possíveis",       node_type: "Concept",  energy: 0.91 },
    Philosopher { name: "Davidson",          school: "Analytic", period: "1917-2003", idea: "Anomalia do mental; verdade e interpretação",    node_type: "Concept",  energy: 0.83 },
    Philosopher { name: "Putnam",            school: "Analytic", period: "1926-2016", idea: "Externalismo semântico; realismo interno",       node_type: "Semantic", energy: 0.85 },
    Philosopher { name: "Rawls",             school: "Analytic", period: "1921-2002", idea: "Teoria da justiça como equidade",                node_type: "Semantic", energy: 0.90 },
    Philosopher { name: "Nozick",            school: "Analytic", period: "1938-2002", idea: "Libertarismo político; anarquia, Estado e utopia", node_type: "Semantic", energy: 0.82 },
    Philosopher { name: "Habermas",          school: "Analytic", period: "1929-hoje", idea: "Teoria da ação comunicativa",                   node_type: "Semantic", energy: 0.88 },
    Philosopher { name: "Chalmers",          school: "Analytic", period: "1966-hoje", idea: "O problema difícil da consciência",              node_type: "Concept",  energy: 0.87 },
    Philosopher { name: "Dennett",           school: "Analytic", period: "1942-hoje", idea: "Heterofenomenologia; consciência explicada",     node_type: "Concept",  energy: 0.86 },
    Philosopher { name: "Searle",            school: "Analytic", period: "1932-hoje", idea: "Intencionalidade e quarto chinês",               node_type: "Concept",  energy: 0.84 },
    Philosopher { name: "Nagel",             school: "Analytic", period: "1937-hoje", idea: "Como é ser um morcego? — a subjetividade",       node_type: "Concept",  energy: 0.85 },
    Philosopher { name: "Parfit",            school: "Analytic", period: "1942-2017", idea: "Razões e pessoas; identidade pessoal",          node_type: "Semantic", energy: 0.83 },
    Philosopher { name: "Williams",          school: "Analytic", period: "1929-2003", idea: "Ética e os limites da filosofia",               node_type: "Semantic", energy: 0.81 },

    // ── E. Nietzsche-core (centro da bola de Poincaré) ────────────────────
    Philosopher { name: "Zaratustra",        school: "Nietzsche-core", period: "mítico",     idea: "O profeta do Übermensch e do eterno retorno",     node_type: "DreamSnapshot", energy: 1.00 },
    Philosopher { name: "Dionísio",          school: "Nietzsche-core", period: "mítico",     idea: "O irracional criativo; força vital do caos",      node_type: "DreamSnapshot", energy: 0.99 },
    Philosopher { name: "Apolo",             school: "Nietzsche-core", period: "mítico",     idea: "Ordem, forma e beleza contra o caos dionisíaco",  node_type: "DreamSnapshot", energy: 0.97 },
    Philosopher { name: "Vontade de Poder",  school: "Nietzsche-core", period: "conceitual", idea: "O impulso fundamental de todo ser vivente",       node_type: "Concept",       energy: 0.98 },
    Philosopher { name: "Eterno Retorno",    school: "Nietzsche-core", period: "conceitual", idea: "Vive cada instante como se fosse eterno",         node_type: "Concept",       energy: 0.96 },
    Philosopher { name: "Übermensch",        school: "Nietzsche-core", period: "conceitual", idea: "O humano que supera a si mesmo",                  node_type: "Concept",       energy: 0.95 },
    Philosopher { name: "Morte de Deus",     school: "Nietzsche-core", period: "conceitual", idea: "Nihilismo e a criação de novos valores",          node_type: "Concept",       energy: 0.94 },
    Philosopher { name: "Ressentimento",     school: "Nietzsche-core", period: "conceitual", idea: "A moral dos escravos e o espírito de vingança",   node_type: "Semantic",      energy: 0.88 },
    Philosopher { name: "Perspectivismo",    school: "Nietzsche-core", period: "conceitual", idea: "Toda verdade é perspectiva; não há fatos, só interpretações", node_type: "Semantic", energy: 0.92 },
    Philosopher { name: "Amor Fati",         school: "Nietzsche-core", period: "conceitual", idea: "Ama teu destino — afirmação total da vida",       node_type: "Semantic",      energy: 0.93 },
    Philosopher { name: "Genealogia da Moral",school:"Nietzsche-core", period: "conceitual", idea: "A origem e história dos valores morais",          node_type: "Concept",       energy: 0.91 },
    Philosopher { name: "Niilismo Ativo",    school: "Nietzsche-core", period: "conceitual", idea: "Destruição criadora de velhos valores",           node_type: "Concept",       energy: 0.90 },
    Philosopher { name: "Niilismo Passivo",  school: "Nietzsche-core", period: "conceitual", idea: "Desespero sem criação de novos valores",          node_type: "Semantic",      energy: 0.60 },
    Philosopher { name: "Grande Saúde",      school: "Nietzsche-core", period: "conceitual", idea: "Vitalidade plena; a saúde que abraça o sofrimento", node_type: "Semantic",    energy: 0.87 },
    Philosopher { name: "Décadence",         school: "Nietzsche-core", period: "conceitual", idea: "Declínio vital e inversão dos valores naturais",  node_type: "Semantic",      energy: 0.55 },
    Philosopher { name: "O Último Homem",    school: "Nietzsche-core", period: "conceitual", idea: "O conformismo medíocre; antítese do Übermensch",  node_type: "Semantic",      energy: 0.45 },
    Philosopher { name: "Inocência do Devir",school: "Nietzsche-core", period: "conceitual", idea: "A criança que brinca; terceira metamorfose",      node_type: "DreamSnapshot", energy: 0.89 },
    Philosopher { name: "O Camelo",          school: "Nietzsche-core", period: "conceitual", idea: "Primeira metamorfose: o espírito que suporta",    node_type: "Episodic",      energy: 0.72 },
    Philosopher { name: "O Leão",            school: "Nietzsche-core", period: "conceitual", idea: "Segunda metamorfose: quero! — liberdade criadora", node_type: "Episodic",     energy: 0.85 },
    Philosopher { name: "A Criança",         school: "Nietzsche-core", period: "conceitual", idea: "Terceira metamorfose: jogo, esquecimento, criação", node_type: "DreamSnapshot", energy: 0.96 },
];

// ── Centros dos clusters no espaço de Poincaré (dim=4) ───────────────────────

fn cluster_center(school: &str) -> [f64; 4] {
    match school {
        "Pre-Socratic"    => [0.12, 0.08, 0.06, 0.05],
        "Idealist"        => [0.08, 0.12, 0.06, 0.05],
        "Existentialist"  => [0.06, 0.08, 0.12, 0.05],
        "Analytic"        => [0.05, 0.06, 0.08, 0.12],
        "Nietzsche-core"  => [0.01, 0.01, 0.01, 0.01],   // centro = origem
        _                 => [0.05, 0.05, 0.05, 0.05],
    }
}

/// Gera coordenadas dentro da bola de Poincaré (||x|| < 1).
/// Adiciona ruído gaussiano ao centro do cluster e normaliza para raio < 0.4.
fn poincare_coords(school: &str, rng: &mut StdRng) -> Vec<f64> {
    let center = cluster_center(school);
    let noise: Vec<f64> = (0..4)
        .map(|i| center[i] + rng.gen_range(-0.08..0.08_f64))
        .collect();

    // Garante ||x|| < 0.4 (dentro da bola de Poincaré)
    let norm: f64 = noise.iter().map(|x| x * x).sum::<f64>().sqrt();
    let max_r = 0.35_f64;
    if norm > max_r {
        noise.iter().map(|x| x * max_r / norm).collect()
    } else {
        noise
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = std::env::var("NIETZSCHE_ADDR")
        .unwrap_or_else(|_| "http://[::1]:50051".to_string());

    println!("Conectando ao NietzscheDB em {addr} ...");
    let mut client = NietzscheClient::connect(addr).await?;

    // Saúde
    let health = client.health_check().await?;
    println!("Health: {}", health.status);

    let mut rng = StdRng::seed_from_u64(42); // seed fixo = reproducível

    // ── 1. Inserir 100 nós ────────────────────────────────────────────────
    println!("\nInserindo 100 nós filosóficos...");
    let mut inserted_ids: Vec<(usize, Uuid)> = Vec::with_capacity(100);

    for (idx, phi) in PHILOSOPHERS.iter().enumerate() {
        let coords = poincare_coords(phi.school, &mut rng);
        let id = Uuid::new_v4();

        let content = json!({
            "name":    phi.name,
            "school":  phi.school,
            "period":  phi.period,
            "idea":    phi.idea,
            "index":   idx,
        });

        let resp = client.insert_node(InsertNodeParams {
            id:        Some(id),
            coords,
            content,
            node_type: phi.node_type.to_string(),
            energy:    phi.energy,
        }).await?;

        println!("  [{:>3}] {} → {} (energy={:.2})",
            idx + 1, phi.name, resp.id, phi.energy);

        inserted_ids.push((idx, id));
    }

    println!("\n✓ {} nós inseridos.", inserted_ids.len());

    // ── 2. Edges intra-cluster em anel ───────────────────────────────────
    println!("\nCriando edges intra-cluster (anel por escola)...");

    let schools = ["Pre-Socratic", "Idealist", "Existentialist", "Analytic", "Nietzsche-core"];
    let mut edge_count = 0u32;

    for school in &schools {
        let school_ids: Vec<Uuid> = inserted_ids.iter()
            .filter(|(idx, _)| PHILOSOPHERS[*idx].school == *school)
            .map(|(_, id)| *id)
            .collect();

        let n = school_ids.len();
        for i in 0..n {
            let from = school_ids[i];
            let to   = school_ids[(i + 1) % n];   // anel circular

            let phi_from = &PHILOSOPHERS[inserted_ids.iter().find(|(_, id)| *id == from).unwrap().0];
            let phi_to   = &PHILOSOPHERS[inserted_ids.iter().find(|(_, id)| *id == to).unwrap().0];

            // Peso = média das energias
            let weight = ((phi_from.energy + phi_to.energy) / 2.0) as f64;

            client.insert_edge(InsertEdgeParams {
                from,
                to,
                edge_type: "Association".into(),
                weight,
                ..Default::default()
            }).await?;
            edge_count += 1;
        }
        println!("  {school}: {n} nodes → {n} edges em anel");
    }

    // ── 3. Edges hierárquicos dentro de cada cluster ──────────────────────
    println!("\nCriando edges hierárquicos (alta energia → baixa energia)...");

    for school in &schools {
        let mut school_nodes: Vec<(Uuid, f32)> = inserted_ids.iter()
            .filter(|(idx, _)| PHILOSOPHERS[*idx].school == *school)
            .map(|(idx, id)| (*id, PHILOSOPHERS[*idx].energy))
            .collect();

        // Ordena por energia decrescente
        school_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Liga top-3 para os seguintes 3 (Hierarchical)
        for i in 0..3.min(school_nodes.len()) {
            for j in (i+1)..(i+4).min(school_nodes.len()) {
                client.insert_edge(InsertEdgeParams {
                    from:      school_nodes[i].0,
                    to:        school_nodes[j].0,
                    edge_type: "Hierarchical".into(),
                    weight:    (school_nodes[i].1 as f64 - school_nodes[j].1 as f64).abs() + 0.1,
                    ..Default::default()
                }).await?;
                edge_count += 1;
            }
        }
    }

    // ── 4. Edges inter-cluster: hub de cada cluster ───────────────────────
    println!("\nCriando edges inter-cluster (hubs)...");

    // Hub = nó com maior energia em cada cluster
    let hubs: Vec<Uuid> = schools.iter().map(|school| {
        inserted_ids.iter()
            .filter(|(idx, _)| PHILOSOPHERS[*idx].school == *school)
            .max_by(|(a, _), (b, _)| {
                PHILOSOPHERS[*a].energy.partial_cmp(&PHILOSOPHERS[*b].energy).unwrap()
            })
            .map(|(_, id)| *id)
            .unwrap()
    }).collect();

    // Liga todos os hubs entre si (grafo completo dos hubs)
    for i in 0..hubs.len() {
        for j in (i+1)..hubs.len() {
            client.insert_edge(InsertEdgeParams {
                from:      hubs[i],
                to:        hubs[j],
                edge_type: "Association".into(),
                weight:    0.5,
                ..Default::default()
            }).await?;
            edge_count += 1;

            // Também no sentido inverso para grafo bidirecional
            client.insert_edge(InsertEdgeParams {
                from:      hubs[j],
                to:        hubs[i],
                edge_type: "Association".into(),
                weight:    0.5,
                ..Default::default()
            }).await?;
            edge_count += 1;
        }
    }

    println!("  {} hubs conectados entre si ({} edges inter-cluster)", hubs.len(), edge_count);

    // ── 5. Stats finais ───────────────────────────────────────────────────
    let stats = client.get_stats().await?;
    println!("\n═══════════════════════════════════════════");
    println!("  Banco de dados NietzscheDB");
    println!("  Nós:   {}", stats.node_count);
    println!("  Edges: {}", stats.edge_count);
    println!("  Versão: {}", stats.version);
    println!("═══════════════════════════════════════════");
    println!("\nSeed completo! {} nós + {} edges inseridos.", inserted_ids.len(), edge_count);
    println!("\nExemplos de queries NQL:");
    println!("  MATCH (n:Concept) WHERE n.energy > 0.9 RETURN n LIMIT 10");
    println!("  MATCH (n) WHERE n.energy > 0.95 RETURN n ORDER BY n.energy DESC LIMIT 5");
    println!("  MATCH (a)-[:Association]->(b) RETURN a, b LIMIT 20");

    Ok(())
}
