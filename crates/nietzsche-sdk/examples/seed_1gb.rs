//! seed_1gb — Carrega ~1 GB de dados filosóficos hierárquicos no NietzscheDB.
//!
//! Gera um grafo de conhecimento filosófico com 5 níveis de profundidade,
//! mapeando a hierarquia para a geometria do Poincaré ball:
//!   centro (norm~0) = conceitos abstratos
//!   borda  (norm~1) = proposições específicas
//!
//! Estrutura:
//!   Nível 0: Raiz (1)
//!   Nível 1: Domínios (8)
//!   Nível 2: Escolas (40)
//!   Nível 3: Pensadores (400)
//!   Nível 4: Conceitos (4,000)
//!   Nível 5: Proposições (~75,551)
//!   Total: ~80,000 nós + ~200,000 edges ≈ 1 GB (at 3072d)
//!
//! Uso:
//!   # Servidor local (padrão)
//!   cargo run -p nietzsche-sdk --release --example seed_1gb
//!
//!   # Servidor remoto + verificação
//!   cargo run -p nietzsche-sdk --release --example seed_1gb -- \
//!     --addr http://10.0.1.5:50051 --verify
//!
//!   # Teste rápido (~10 MB, 64d)
//!   cargo run -p nietzsche-sdk --example seed_1gb -- --target-mb 10 --dim 64

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nietzsche_sdk::{InsertEdgeParams, InsertNodeParams, NietzscheClient};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Parser)]
#[command(name = "seed_1gb")]
#[command(about = "Carrega ~1 GB de dados filosóficos hierárquicos no NietzscheDB")]
struct Cli {
    /// Endereço gRPC do NietzscheDB
    #[arg(long, env = "NIETZSCHE_ADDR", default_value = "http://[::1]:50051")]
    addr: String,

    /// Tamanho alvo em megabytes
    #[arg(long, default_value = "1024")]
    target_mb: u64,

    /// Dimensão dos embeddings
    #[arg(long, default_value = "3072")]
    dim: usize,

    /// Conexões gRPC paralelas
    #[arg(long, default_value = "16")]
    workers: usize,

    /// Seed RNG para reprodutibilidade
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Collection alvo (vazio = "default")
    #[arg(long, default_value = "")]
    collection: String,

    /// Pular geração de edges
    #[arg(long)]
    nodes_only: bool,

    /// Verificar contagens após inserção
    #[arg(long)]
    verify: bool,

    /// Apenas calcular e exibir o plano (sem conectar ao servidor)
    #[arg(long)]
    dry_run: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DADOS FILOSÓFICOS ESTÁTICOS
// ═══════════════════════════════════════════════════════════════════════════════

const DOMAINS: &[(&str, &str)] = &[
    ("Metafísica", "Estudo do ser enquanto ser e da realidade última"),
    ("Epistemologia", "Teoria do conhecimento, justificação e crença"),
    ("Ética", "Investigação sobre o bem, o dever e a virtude"),
    ("Estética", "Filosofia da arte, beleza e experiência sensível"),
    ("Lógica", "Estrutura do raciocínio válido e da inferência"),
    ("Filosofia Política", "Justiça, poder, Estado e liberdade"),
    ("Filosofia da Mente", "Consciência, intencionalidade e relação mente-corpo"),
    ("Filosofia da Linguagem", "Significado, referência e uso da linguagem"),
];

const SCHOOLS: &[&[&str]] = &[
    // Metafísica
    &["Racionalismo", "Empirismo", "Idealismo", "Materialismo", "Dualismo"],
    // Epistemologia
    &["Fundacionalismo", "Coerentismo", "Ceticismo", "Pragmatismo", "Naturalismo"],
    // Ética
    &["Deontologia", "Consequencialismo", "Ética das Virtudes", "Contratualismo", "Existencialismo Ético"],
    // Estética
    &["Formalismo", "Expressionismo", "Institucionalismo", "Fenomenologia Estética", "Pós-modernismo"],
    // Lógica
    &["Lógica Clássica", "Lógica Modal", "Lógica Intuicionista", "Lógica Paraconsistente", "Lógica Fuzzy"],
    // Filosofia Política
    &["Liberalismo", "Comunitarismo", "Marxismo", "Anarquismo", "Republicanismo"],
    // Filosofia da Mente
    &["Funcionalismo", "Dualismo de Propriedades", "Eliminativismo", "Panpsiquismo", "Emergentismo"],
    // Filosofia da Linguagem
    &["Referencialismo", "Descritivismo", "Pragmática", "Estruturalismo", "Hermenêutica"],
];

// 400 pensadores (10 por escola × 40 escolas)
// Organizados por domínio → escola
const THINKERS: &[&[&[&str]]] = &[
    // Metafísica: Racionalismo, Empirismo, Idealismo, Materialismo, Dualismo
    &[
        &["Descartes", "Spinoza", "Leibniz", "Wolff", "Malebranche", "Arnauld", "Gassendi", "Suárez", "Duns Scotus", "Ockham"],
        &["Locke", "Hume", "Berkeley", "Bacon", "Hobbes", "Condillac", "Mill J.S.", "Reid", "Helmholtz", "Mach"],
        &["Platão", "Hegel", "Fichte", "Schelling", "Bradley", "McTaggart", "Gentile", "Croce", "Royce", "Bosanquet"],
        &["Demócrito", "Epicuro", "Lucrécio", "La Mettrie", "d'Holbach", "Feuerbach", "Büchner", "Vogt", "Moleschott", "Smart"],
        &["Aristóteles", "Tomás de Aquino", "Avicena", "Popper", "Chalmers", "Eccles", "Swinburne", "Hasker", "Lowe", "Foster"],
    ],
    // Epistemologia
    &[
        &["Descartes E.", "Leibniz E.", "BonJour", "Chisholm", "Plantinga", "Fumerton", "McGrew", "Huemer", "Audi", "Pryor"],
        &["Quine", "Sellars", "Davidson", "Harman", "Lehrer", "Rescher", "Thagard", "Young", "Olsson", "Bovens"],
        &["Pirro", "Sexto Empírico", "Montaigne", "Descartes Cético", "Hume Cético", "Unger", "Stroud", "Fogelin", "McGinn", "Pritchard"],
        &["Peirce", "James", "Dewey", "Rorty", "Putnam H.", "Brandom", "Misak", "Hookway", "Haack", "Rescher P."],
        &["Kornblith", "Goldman", "Kitcher", "Laudan", "Stich", "Bishop", "Trout", "Gigerenzer", "Kahneman", "Stanovich"],
    ],
    // Ética
    &[
        &["Kant", "Ross", "Korsgaard", "O'Neill", "Herman", "Wood", "Scanlon", "Parfit", "Darwall", "Nagel T."],
        &["Bentham", "Mill", "Sidgwick", "Singer", "Hare", "Railton", "Brink", "Scheffler", "Kagan", "Sinnott-Armstrong"],
        &["Aristóteles É.", "MacIntyre", "Foot", "Hursthouse", "Annas", "Swanton", "Slote", "Zagzebski", "Russell D.", "Snow"],
        &["Hobbes É.", "Locke É.", "Rousseau", "Rawls", "Gauthier", "Scanlon T.", "Nussbaum", "Sen", "Freeman", "Barry"],
        &["Kierkegaard", "Nietzsche", "Sartre É.", "Camus É.", "Beauvoir", "Marcel", "Buber É.", "Levinas É.", "Jaspers É.", "Tillich"],
    ],
    // Estética
    &[
        &["Kant Est.", "Bell", "Greenberg", "Hanslick", "Sibley", "Zangwill", "Carroll N.", "Lopes", "Shelley", "Levinson"],
        &["Tolstoy", "Collingwood", "Croce Est.", "Dewey Est.", "Langer", "Goodman", "Robinson J.", "Davies S.", "Kivy", "Matravers"],
        &["Dickie", "Danto", "Becker", "Wollheim", "Walton", "Currie", "Stecker", "Lamarque", "Olsen", "Gaut"],
        &["Heidegger Est.", "Merleau-Ponty Est.", "Dufrenne", "Ingarden", "Geiger", "Hartmann N.", "Lévinas Est.", "Henry", "Barbaras", "Romano"],
        &["Lyotard Est.", "Baudrillard Est.", "Jameson", "Foster H.", "Krauss", "Rancière", "Bourriaud", "Groys", "Virilio", "Stiegler"],
    ],
    // Lógica
    &[
        &["Frege", "Russell", "Whitehead", "Gödel", "Tarski", "Church", "Turing", "Hilbert", "Peano", "Boole"],
        &["Kripke", "Lewis D.", "Carnap L.", "von Wright", "Hintikka", "Hughes", "Cresswell", "Fitting", "Blackburn", "Chellas"],
        &["Brouwer", "Heyting", "Dummett", "Martin-Löf", "Bishop M.", "Troelstra", "van Dalen", "Bridges", "Richman", "Beeson"],
        &["da Costa", "Priest", "Routley", "Belnap", "Dunn", "Asenjo", "Batens", "Carnielli", "Marcos", "Avron"],
        &["Zadeh", "Goguen", "Hájek", "Novák", "Esteva", "Godo", "Cintula", "Bĕhounek", "Fermüller", "Metcalfe"],
    ],
    // Filosofia Política
    &[
        &["Locke P.", "Mill P.", "Rawls P.", "Dworkin", "Nozick", "Hayek", "Berlin", "Raz", "Kymlicka", "Gaus"],
        &["Taylor Ch.", "Sandel", "Walzer", "MacIntyre P.", "Etzioni", "Bellah", "Selznick", "Tam", "Frazer E.", "Miller D."],
        &["Marx", "Engels", "Gramsci", "Lukács", "Adorno", "Horkheimer", "Marcuse", "Althusser", "Poulantzas", "Harvey"],
        &["Bakunin", "Kropotkin", "Proudhon", "Goldman E.", "Malatesta", "Bookchin", "Chomsky P.", "Graeber", "Scott", "Ward"],
        &["Maquiavel", "Cícero", "Arendt P.", "Pettit", "Skinner Q.", "Pocock", "Viroli", "Laborde", "Lovett", "Honohan"],
    ],
    // Filosofia da Mente
    &[
        &["Putnam FM.", "Fodor", "Block", "Armstrong", "Lewis FM.", "Lycan", "Shoemaker", "Kim", "McLaughlin", "Horgan"],
        &["Jackson", "Nagel FM.", "Chalmers FM.", "Levine", "McGinn FM.", "Robinson H.", "Nida-Rümelin", "Alter", "Walter", "Stoljar"],
        &["Churchland P.", "Churchland P.S.", "Stich FM.", "Ramsey", "Bickle", "Mandik", "Hardcastle", "Machery", "Mallon", "Piccinini"],
        &["Chalmers Pan.", "Strawson G.", "Nagel Pan.", "Seager", "Skrbina", "Brüntrup", "Jaskolla", "Koch", "Tononi", "Mørch"],
        &["Mill Em.", "Alexander S.", "Morgan C.L.", "Sperry", "O'Connor", "Wong", "Clayton", "Bedau", "Humphreys", "Silberstein"],
    ],
    // Filosofia da Linguagem
    &[
        &["Frege FL.", "Russell FL.", "Kripke FL.", "Kaplan", "Donnellan", "Putnam FL.", "Burge", "Salmon", "Soames", "Devitt"],
        &["Frege D.", "Russell D.", "Searle FL.", "Strawson", "Evans", "Jackson FL.", "Lewis FL.", "Chalmers FL.", "Kroon", "Recanati"],
        &["Austin", "Grice", "Levinson", "Sperber", "Wilson D.", "Carston", "Bach", "Horn", "Huang", "Noveck"],
        &["Saussure", "Jakobson", "Lévi-Strauss", "Barthes", "Derrida FL.", "Foucault FL.", "Kristeva", "Eco", "Greimas", "Benveniste"],
        &["Schleiermacher", "Dilthey", "Heidegger H.", "Gadamer", "Ricoeur", "Habermas H.", "Apel", "Vattimo", "Rorty H.", "Taylor H."],
    ],
];

const KEY_IDEAS: &[&str] = &[
    "A substância como fundamento do real",
    "A experiência como fonte primária do conhecimento",
    "A forma como essência inteligível das coisas",
    "A matéria como única realidade existente",
    "A dualidade irredutível entre mente e corpo",
    "A verdade como correspondência com os fatos",
    "A coerência interna como critério de verdade",
    "A suspensão do juízo como caminho para a paz",
    "A utilidade prática como medida do conhecimento",
    "A evolução como motor do progresso cognitivo",
    "O dever como imperativo incondicional",
    "A maximização da felicidade como fim moral",
    "A excelência do caráter como vida boa",
    "O contrato social como base da obrigação",
    "A autenticidade como valor supremo",
    "A beleza como harmonia de forma e conteúdo",
    "A expressão como essência da arte",
    "O contexto institucional como definidor da arte",
    "A percepção como encontro com o ser",
    "A fragmentação como condição contemporânea",
    "A validade lógica como preservação de verdade",
    "A necessidade e possibilidade como modalidades",
    "A construtividade como requisito de existência",
    "A contradição como aspecto do real",
    "A gradualidade como natureza da verdade",
    "A liberdade individual como valor fundamental",
    "A comunidade como horizonte de significado",
    "A luta de classes como motor da história",
    "A autogestão como organização ideal",
    "A participação cívica como liberdade",
    "A consciência como fenômeno irredutível",
    "A mente como programa computacional",
    "A eliminação de conceitos folk da ciência",
    "A experiência como propriedade fundamental",
    "A emergência como novidade ontológica",
    "A referência direta aos objetos do mundo",
    "A descrição como mecanismo de fixação",
    "O uso como determinante do significado",
    "A estrutura como sistema de diferenças",
    "A interpretação como fusão de horizontes",
];

const CONCEPT_TEMPLATES: &[&str] = &[
    "Teoria de {}", "Crítica de {}", "Fundamentos de {}", "Paradoxo de {}",
    "Princípio de {}", "Método de {}", "Dialética de {}", "Fenomenologia de {}",
    "Análise de {}", "Síntese de {}", "Genealogia de {}", "Hermenêutica de {}",
    "Ontologia de {}", "Axioma de {}", "Aporia de {}", "Telos de {}",
    "Pragma de {}", "Logos de {}", "Ethos de {}", "Pathos de {}",
];

const PROPOSITION_PREFIXES: &[&str] = &[
    "Argumento", "Tese", "Demonstração", "Refutação", "Objeção",
    "Corolário", "Lema", "Proposição", "Teorema", "Hipótese",
    "Axioma", "Definição", "Postulado", "Inferência", "Dedução",
    "Indução", "Abdução", "Analogia", "Contra-exemplo", "Paradoxo",
];

const PROPOSITION_SUFFIXES: &[&str] = &[
    "na tradição ocidental",
    "segundo a escola analítica",
    "na perspectiva continental",
    "à luz da fenomenologia",
    "conforme a dialética hegeliana",
    "via análise linguística",
    "pelo método transcendental",
    "sob a ótica pragmatista",
    "contra o ceticismo radical",
    "em defesa do realismo",
    "na crítica pós-moderna",
    "pela hermenêutica filosófica",
    "segundo o naturalismo",
    "na tradição oriental comparada",
    "pelo existencialismo",
    "conforme a teoria crítica",
    "via lógica modal",
    "sob a perspectiva ética",
    "contra o reducionismo",
    "em favor do pluralismo",
];

const WORKS: &[&str] = &[
    "Crítica da Razão Pura", "Ética a Nicômaco", "Meditações Metafísicas",
    "Ser e Tempo", "O Ser e o Nada", "Fenomenologia do Espírito",
    "Investigações Filosóficas", "Tractatus Logico-Philosophicus",
    "A República", "O Príncipe", "Leviatã", "Ensaio Sobre o Entendimento Humano",
    "Tratado da Natureza Humana", "Genealogia da Moral",
    "Assim Falou Zaratustra", "O Capital", "Uma Teoria da Justiça",
    "Verdade e Método", "A Condição Humana", "Vigiar e Punir",
    "A Estrutura das Revoluções Científicas", "Dois Dogmas do Empirismo",
    "Principia Mathematica", "Da Certeza", "Lógica da Descoberta Científica",
    "Totalidade e Infinito", "O Mito de Sísifo", "O Segundo Sexo",
    "Mil Platôs", "A Sociedade do Espetáculo",
];

const LANGUAGES: &[&str] = &[
    "Grego", "Latim", "Alemão", "Francês", "Inglês",
    "Árabe", "Chinês", "Sânscrito", "Português", "Italiano",
];

const ERAS: &[&str] = &[
    "Antiguidade", "Período Clássico", "Helenismo", "Idade Média",
    "Renascimento", "Modernidade", "Iluminismo", "Romantismo",
    "Século XIX", "Contemporâneo",
];

const REGIONS: &[&str] = &[
    "Grécia", "Roma", "Alemanha", "França", "Inglaterra",
    "Pérsia", "China", "Índia", "Estados Unidos", "Itália",
];

// ═══════════════════════════════════════════════════════════════════════════════
// ONTOLOGIA
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct OntologyNode {
    level:        u8,
    parent_idx:   Option<usize>,
    label:        String,
    domain_idx:   usize,
    school_idx:   usize,
    thinker_idx:  usize,
    node_type:    &'static str,
    energy:       f32,
    // Embedding direction (unit vector) for generating children
    direction:    Vec<f64>,
}

fn compute_node_count(target_mb: u64, dim: usize) -> usize {
    let bytes_per_node = (dim * 4) + 300 + 200; // f32 coords + content + overhead
    let bytes_per_edge: f64 = 100.0;
    let edges_per_node: f64 = 2.5;
    let total_per_node = bytes_per_node as f64 + edges_per_node * bytes_per_edge;
    let target_bytes = target_mb as f64 * 1024.0 * 1024.0;
    (target_bytes / total_per_node) as usize
}

fn build_ontology(total_nodes: usize, dim: usize, rng: &mut StdRng) -> Vec<OntologyNode> {
    let mut nodes = Vec::with_capacity(total_nodes);

    // --- Nível 0: Raiz ---
    let root_dir = random_direction(dim, rng);
    nodes.push(OntologyNode {
        level: 0,
        parent_idx: None,
        label: "Filosofia".into(),
        domain_idx: 0,
        school_idx: 0,
        thinker_idx: 0,
        node_type: "Concept",
        energy: 1.0,
        direction: root_dir,
    });

    // --- Nível 1: Domínios (8) ---
    let domain_start = nodes.len();
    for (di, (name, _desc)) in DOMAINS.iter().enumerate() {
        let dir = child_direction(&nodes[0].direction, 0.5, rng);
        nodes.push(OntologyNode {
            level: 1,
            parent_idx: Some(0),
            label: name.to_string(),
            domain_idx: di,
            school_idx: 0,
            thinker_idx: 0,
            node_type: "Concept",
            energy: 0.95 + rng.gen_range(0.0..0.05_f32),
            direction: dir,
        });
    }

    // --- Nível 2: Escolas (40 = 5×8) ---
    let school_start = nodes.len();
    for di in 0..DOMAINS.len() {
        let parent_idx = domain_start + di;
        for (si, school_name) in SCHOOLS[di].iter().enumerate() {
            let dir = child_direction(&nodes[parent_idx].direction, 0.3, rng);
            nodes.push(OntologyNode {
                level: 2,
                parent_idx: Some(parent_idx),
                label: school_name.to_string(),
                domain_idx: di,
                school_idx: si,
                thinker_idx: 0,
                node_type: "Concept",
                energy: 0.85 + rng.gen_range(0.0..0.10_f32),
                direction: dir,
            });
        }
    }

    // --- Nível 3: Pensadores (400 = 10×40) ---
    let thinker_start = nodes.len();
    for di in 0..DOMAINS.len() {
        for si in 0..5 {
            let parent_idx = school_start + di * 5 + si;
            for (ti, thinker_name) in THINKERS[di][si].iter().enumerate() {
                let dir = child_direction(&nodes[parent_idx].direction, 0.2, rng);
                nodes.push(OntologyNode {
                    level: 3,
                    parent_idx: Some(parent_idx),
                    label: thinker_name.to_string(),
                    domain_idx: di,
                    school_idx: si,
                    thinker_idx: ti,
                    node_type: "Semantic",
                    energy: 0.70 + rng.gen_range(0.0..0.20_f32),
                    direction: dir,
                });
            }
        }
    }

    // --- Nível 4: Conceitos (4,000 = 10×400) ---
    let concept_start = nodes.len();
    for ti_global in 0..400 {
        let parent_idx = thinker_start + ti_global;
        let thinker_label = nodes[parent_idx].label.clone();
        for ci in 0..10 {
            let template = CONCEPT_TEMPLATES[ci % CONCEPT_TEMPLATES.len()];
            let label = template.replace("{}", &thinker_label);
            let dir = child_direction(&nodes[parent_idx].direction, 0.1, rng);
            nodes.push(OntologyNode {
                level: 4,
                parent_idx: Some(parent_idx),
                label,
                domain_idx: nodes[parent_idx].domain_idx,
                school_idx: nodes[parent_idx].school_idx,
                thinker_idx: ti_global,
                node_type: "Concept",
                energy: 0.50 + rng.gen_range(0.0..0.35_f32),
                direction: dir,
            });
        }
    }

    // --- Nível 5: Proposições (restante até total_nodes) ---
    let remaining = total_nodes.saturating_sub(nodes.len());
    let num_concepts = 4000.min(concept_start + 4000 - concept_start);
    let per_concept = if num_concepts > 0 { remaining / num_concepts } else { 0 };
    let mut extra = remaining - per_concept * num_concepts;

    let node_types_l5 = ["Semantic", "Semantic", "Semantic", "Episodic", "Concept", "DreamSnapshot"];

    for ci_global in 0..num_concepts {
        let parent_idx = concept_start + ci_global;
        let concept_label = nodes[parent_idx].label.clone();
        let count = per_concept + if extra > 0 { extra -= 1; 1 } else { 0 };

        for pi in 0..count {
            let prefix = PROPOSITION_PREFIXES[(ci_global * 31 + pi) % PROPOSITION_PREFIXES.len()];
            let suffix = PROPOSITION_SUFFIXES[(ci_global * 17 + pi) % PROPOSITION_SUFFIXES.len()];
            let label = format!("{prefix}: {concept_label} — {suffix}");
            let dir = child_direction(&nodes[parent_idx].direction, 0.05, rng);
            let nt_idx = (ci_global + pi) % node_types_l5.len();
            nodes.push(OntologyNode {
                level: 5,
                parent_idx: Some(parent_idx),
                label,
                domain_idx: nodes[parent_idx].domain_idx,
                school_idx: nodes[parent_idx].school_idx,
                thinker_idx: nodes[parent_idx].thinker_idx,
                node_type: node_types_l5[nt_idx],
                energy: 0.30 + rng.gen_range(0.0..0.50_f32),
                direction: dir,
            });
        }
    }

    // Trim to exact target
    nodes.truncate(total_nodes);
    nodes
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMBEDDING GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

fn random_direction(dim: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0_f64)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn child_direction(parent_dir: &[f64], angular_spread: f64, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = parent_dir
        .iter()
        .map(|&d| d + rng.gen_range(-angular_spread..angular_spread))
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn target_radius(level: u8, rng: &mut StdRng) -> f64 {
    let (base, jitter) = match level {
        0 => (0.02, 0.01),
        1 => (0.10, 0.02),
        2 => (0.30, 0.05),
        3 => (0.55, 0.05),
        4 => (0.75, 0.05),
        _ => (0.92, 0.03),
    };
    (base + rng.gen_range(-jitter..jitter) as f64).clamp(0.01, 0.995)
}

fn generate_embedding(direction: &[f64], radius: f64) -> Vec<f64> {
    direction.iter().map(|&d| d * radius).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_content(node: &OntologyNode, idx: usize) -> serde_json::Value {
    match node.level {
        0 => json!({
            "name": node.label,
            "type": "root",
            "description": "Raiz do grafo de conhecimento filosófico",
        }),
        1 => json!({
            "name": node.label,
            "type": "domain",
            "description": DOMAINS[node.domain_idx].1,
            "index": idx,
        }),
        2 => json!({
            "name": node.label,
            "type": "school",
            "domain": DOMAINS[node.domain_idx].0,
            "era": ERAS[node.school_idx % ERAS.len()],
            "region": REGIONS[node.domain_idx % REGIONS.len()],
        }),
        3 => json!({
            "name": node.label,
            "type": "thinker",
            "school": SCHOOLS[node.domain_idx][node.school_idx],
            "domain": DOMAINS[node.domain_idx].0,
            "key_idea": KEY_IDEAS[idx % KEY_IDEAS.len()],
            "period": ERAS[(idx * 3) % ERAS.len()],
            "region": REGIONS[(idx * 7) % REGIONS.len()],
        }),
        4 => json!({
            "name": node.label,
            "type": "concept",
            "domain": DOMAINS[node.domain_idx].0,
            "description": KEY_IDEAS[idx % KEY_IDEAS.len()],
        }),
        _ => json!({
            "text": node.label,
            "type": "proposition",
            "source_work": WORKS[idx % WORKS.len()],
            "year": 300 + (idx % 2200),
            "language": LANGUAGES[idx % LANGUAGES.len()],
            "confidence": ((idx as f64 * 0.37).fract() * 100.0).round() / 100.0,
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_edges(
    ontology: &[OntologyNode],
    uuids: &[Uuid],
    rng: &mut StdRng,
    target_edges: usize,
) -> Vec<InsertEdgeParams> {
    let mut edges = Vec::with_capacity(target_edges);

    // Pass 1: Hierarchical tree edges (parent → child)
    for (i, node) in ontology.iter().enumerate() {
        if let Some(pi) = node.parent_idx {
            edges.push(InsertEdgeParams {
                from: uuids[pi],
                to: uuids[i],
                edge_type: "Hierarchical".into(),
                weight: 0.8 + rng.gen_range(0.0..0.2_f64),
                ..Default::default()
            });
        }
    }

    // Pass 2: Sibling associations (same parent)
    let mut children_of: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (i, node) in ontology.iter().enumerate() {
        if let Some(pi) = node.parent_idx {
            children_of.entry(pi).or_default().push(i);
        }
    }
    for (_parent, children) in &children_of {
        let n = children.len();
        let pairs = 5.min(n / 2);
        for _ in 0..pairs {
            let a = children[rng.gen_range(0..n)];
            let b = children[rng.gen_range(0..n)];
            if a != b {
                edges.push(InsertEdgeParams {
                    from: uuids[a],
                    to: uuids[b],
                    edge_type: "Association".into(),
                    weight: 0.5 + rng.gen_range(0.0..0.4_f64),
                    ..Default::default()
                });
            }
        }
    }

    // Pass 3: Cross-school links (same domain, different school, levels 3-4)
    let cross_count = (target_edges / 10).min(20_000);
    for _ in 0..cross_count {
        let a = rng.gen_range(0..ontology.len());
        let b = rng.gen_range(0..ontology.len());
        if a != b
            && ontology[a].level >= 3
            && ontology[b].level >= 3
            && ontology[a].domain_idx == ontology[b].domain_idx
            && ontology[a].school_idx != ontology[b].school_idx
        {
            edges.push(InsertEdgeParams {
                from: uuids[a],
                to: uuids[b],
                edge_type: "Association".into(),
                weight: 0.3 + rng.gen_range(0.0..0.4_f64),
                ..Default::default()
            });
        }
    }

    // Pass 4: Cross-domain analogies
    let analogy_count = (target_edges / 20).min(10_000);
    for _ in 0..analogy_count {
        let a = rng.gen_range(0..ontology.len());
        let b = rng.gen_range(0..ontology.len());
        if a != b
            && ontology[a].level >= 2
            && ontology[b].level >= 2
            && ontology[a].level <= 4
            && ontology[b].level <= 4
            && ontology[a].domain_idx != ontology[b].domain_idx
        {
            edges.push(InsertEdgeParams {
                from: uuids[a],
                to: uuids[b],
                edge_type: "LSystemGenerated".into(),
                weight: 0.1 + rng.gen_range(0.0..0.3_f64),
                ..Default::default()
            });
        }
    }

    // Trim to target
    edges.truncate(target_edges);
    edges
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKER POOL
// ═══════════════════════════════════════════════════════════════════════════════

enum WorkItem {
    Node { idx: usize, params: InsertNodeParams },
    Edge(InsertEdgeParams),
}

async fn spawn_node_workers(
    addr: String,
    n_workers: usize,
    rx: Arc<tokio::sync::Mutex<mpsc::Receiver<WorkItem>>>,
    uuid_map: Arc<tokio::sync::Mutex<Vec<Uuid>>>,
    pb: ProgressBar,
) -> u64 {
    let mut handles = Vec::with_capacity(n_workers);

    for _w in 0..n_workers {
        let addr = addr.clone();
        let rx = Arc::clone(&rx);
        let uuid_map = Arc::clone(&uuid_map);
        let pb = pb.clone();

        handles.push(tokio::spawn(async move {
            let mut client = match NietzscheClient::connect(&addr).await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Worker falhou ao conectar: {e}");
                    return 0u64;
                }
            };

            let mut count = 0u64;
            loop {
                let item = {
                    let mut guard = rx.lock().await;
                    guard.recv().await
                };
                match item {
                    Some(WorkItem::Node { idx, params }) => {
                        let id = params.id.unwrap_or_else(Uuid::new_v4);
                        let mut retries = 0;
                        loop {
                            match client.insert_node(params.clone()).await {
                                Ok(_resp) => {
                                    let mut map = uuid_map.lock().await;
                                    if idx < map.len() {
                                        map[idx] = id;
                                    }
                                    count += 1;
                                    pb.inc(1);
                                    break;
                                }
                                Err(e) => {
                                    retries += 1;
                                    if retries >= 3 {
                                        eprintln!("Falha ao inserir nó {idx}: {e}");
                                        pb.inc(1);
                                        break;
                                    }
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        100 * (1 << retries),
                                    ))
                                    .await;
                                }
                            }
                        }
                    }
                    Some(WorkItem::Edge(params)) => {
                        let mut retries = 0;
                        loop {
                            match client.insert_edge(params.clone()).await {
                                Ok(_) => {
                                    count += 1;
                                    pb.inc(1);
                                    break;
                                }
                                Err(e) => {
                                    retries += 1;
                                    if retries >= 3 {
                                        eprintln!("Falha ao inserir edge: {e}");
                                        pb.inc(1);
                                        break;
                                    }
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        100 * (1 << retries),
                                    ))
                                    .await;
                                }
                            }
                        }
                    }
                    None => break, // channel closed
                }
            }
            count
        }));
    }

    let mut total = 0u64;
    for h in handles {
        total += h.await.unwrap_or(0);
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    let start = Instant::now();

    // ── Calcular volumes ──
    let total_nodes = compute_node_count(cli.target_mb, cli.dim);
    let total_edges = if cli.nodes_only { 0 } else { (total_nodes as f64 * 2.5) as usize };
    let est_size_mb = (total_nodes as f64 * (cli.dim as f64 * 4.0 + 300.0 + 200.0)
        + total_edges as f64 * 100.0)
        / (1024.0 * 1024.0);

    println!("\n== Plano de Carga ==");
    println!("  Target:      {} MB", cli.target_mb);
    println!("  Dimensão:    {}d", cli.dim);
    println!("  Nós:         {}", total_nodes);
    println!("  Edges:       {}", total_edges);
    println!("  Est. disco:  {:.0} MB", est_size_mb);
    println!("  Workers:     {}", cli.workers);
    println!("  Seed RNG:    {}", cli.seed);
    println!();

    if cli.dry_run {
        println!("(--dry-run) Calculando ontologia sem conectar...");
        let mut rng = StdRng::seed_from_u64(cli.seed);
        let ontology = build_ontology(total_nodes, cli.dim, &mut rng);
        let level_counts: Vec<usize> = (0..=5)
            .map(|l| ontology.iter().filter(|n| n.level == l).count())
            .collect();
        println!(
            "  L0:{} L1:{} L2:{} L3:{} L4:{} L5:{}",
            level_counts[0], level_counts[1], level_counts[2],
            level_counts[3], level_counts[4], level_counts[5]
        );
        println!("  Total nós: {}", ontology.len());
        println!("  Total edges estimado: {}", total_edges);
        println!("\nDry run completo. Use sem --dry-run para executar a carga.");
        return Ok(());
    }

    // ── Health check ──
    println!("Conectando ao NietzscheDB em {} ...", cli.addr);
    let mut check_client = NietzscheClient::connect(&cli.addr).await?;
    let health = check_client.health_check().await?;
    println!("Health: {}", health.status);
    drop(check_client);

    // ── Construir ontologia ──
    let mut rng = StdRng::seed_from_u64(cli.seed);
    println!("Construindo ontologia hierárquica...");
    let ontology = build_ontology(total_nodes, cli.dim, &mut rng);
    println!("  {} nós no plano ontológico", ontology.len());

    let level_counts: Vec<usize> = (0..=5)
        .map(|l| ontology.iter().filter(|n| n.level == l).count())
        .collect();
    println!(
        "  L0:{} L1:{} L2:{} L3:{} L4:{} L5:{}",
        level_counts[0], level_counts[1], level_counts[2],
        level_counts[3], level_counts[4], level_counts[5]
    );

    // ── Fase 1: Inserir nós ──
    let mp = MultiProgress::new();
    let sty = ProgressStyle::with_template(
        "  {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) ETA {eta}",
    )
    .unwrap()
    .progress_chars("=>-");

    println!("\nFase 1/2: Inserindo {} nós...", ontology.len());
    let pb_nodes = mp.add(ProgressBar::new(ontology.len() as u64));
    pb_nodes.set_style(sty.clone());

    let uuid_map = Arc::new(tokio::sync::Mutex::new(vec![Uuid::nil(); ontology.len()]));
    let (tx, rx) = mpsc::channel::<WorkItem>(1000);
    let rx = Arc::new(tokio::sync::Mutex::new(rx));

    // Spawn workers
    let worker_handle = {
        let addr = cli.addr.clone();
        let uuid_map = Arc::clone(&uuid_map);
        let pb = pb_nodes.clone();
        let n_workers = cli.workers;
        let rx = Arc::clone(&rx);
        tokio::spawn(async move { spawn_node_workers(addr, n_workers, rx, uuid_map, pb).await })
    };

    // Generate and send nodes
    let mut emb_rng = StdRng::seed_from_u64(cli.seed + 1);
    for (idx, node) in ontology.iter().enumerate() {
        let radius = target_radius(node.level, &mut emb_rng);
        let coords = generate_embedding(&node.direction, radius);
        let id = Uuid::new_v4();
        let content = generate_content(node, idx);

        // Store the ID ahead of time for edge generation
        {
            let mut map = uuid_map.lock().await;
            map[idx] = id;
        }

        let params = InsertNodeParams {
            id: Some(id),
            coords,
            content,
            node_type: node.node_type.to_string(),
            energy: node.energy,
        };

        tx.send(WorkItem::Node { idx, params }).await?;
    }
    drop(tx); // signal workers to finish

    let nodes_inserted = worker_handle.await?;
    pb_nodes.finish_with_message("completo");
    println!("  {} nós inseridos em {:.1}s", nodes_inserted, start.elapsed().as_secs_f64());

    // ── Fase 2: Inserir edges ──
    if !cli.nodes_only && total_edges > 0 {
        println!("\nFase 2/2: Gerando e inserindo {} edges...", total_edges);
        let pb_edges = mp.add(ProgressBar::new(total_edges as u64));
        pb_edges.set_style(sty);

        let uuids = uuid_map.lock().await.clone();
        let mut edge_rng = StdRng::seed_from_u64(cli.seed + 2);
        let edges = generate_edges(&ontology, &uuids, &mut edge_rng, total_edges);
        let actual_edges = edges.len();
        pb_edges.set_length(actual_edges as u64);

        let (tx_e, rx_e) = mpsc::channel::<WorkItem>(1000);
        let rx_e = Arc::new(tokio::sync::Mutex::new(rx_e));

        let edge_worker_handle = {
            let addr = cli.addr.clone();
            let uuid_map = Arc::clone(&uuid_map);
            let pb = pb_edges.clone();
            let n_workers = cli.workers;
            tokio::spawn(async move {
                spawn_node_workers(addr, n_workers, rx_e, uuid_map, pb).await
            })
        };

        for edge in edges {
            tx_e.send(WorkItem::Edge(edge)).await?;
        }
        drop(tx_e);

        let edges_inserted = edge_worker_handle.await?;
        pb_edges.finish_with_message("completo");
        println!(
            "  {} edges inseridos em {:.1}s",
            edges_inserted,
            start.elapsed().as_secs_f64()
        );
    }

    // ── Verificação ──
    if cli.verify {
        println!("\nVerificando...");
        let mut v_client = NietzscheClient::connect(&cli.addr).await?;
        let stats = v_client.get_stats().await?;
        println!("  Nós no servidor:   {}", stats.node_count);
        println!("  Edges no servidor: {}", stats.edge_count);
        println!("  Versão:            {}", stats.version);

        // Amostragem: pegar 5 nós aleatórios
        let uuids = uuid_map.lock().await;
        let mut sample_rng = StdRng::seed_from_u64(cli.seed + 99);
        let mut ok = 0;
        for _ in 0..5 {
            let idx = sample_rng.gen_range(0..uuids.len());
            let uid = uuids[idx];
            if uid.is_nil() {
                continue;
            }
            match v_client.get_node(uid).await {
                Ok(resp) => {
                    if let Some(emb) = resp.embedding {
                        let norm_sq: f64 = emb.coords.iter().map(|x| x * x).sum();
                        if norm_sq < 1.0 {
                            ok += 1;
                        } else {
                            eprintln!("  WARN: nó {} tem norma {:.4} >= 1.0", uid, norm_sq.sqrt());
                        }
                    }
                }
                Err(e) => eprintln!("  WARN: falha ao buscar nó {}: {}", uid, e),
            }
        }
        println!("  Amostragem: {}/5 nós OK (norma < 1.0)", ok);
    }

    // ── Resumo ──
    let elapsed = start.elapsed();
    println!("\n═══════════════════════════════════════════");
    println!("  seed_1gb — Carga completa!");
    println!("  Nós:         {}", total_nodes);
    println!("  Edges:       {}", total_edges);
    println!("  Dimensão:    {}d", cli.dim);
    println!("  Est. disco:  {:.0} MB", est_size_mb);
    println!("  Tempo total: {:.1}s", elapsed.as_secs_f64());
    if elapsed.as_secs() > 0 {
        println!(
            "  Throughput:  {:.0} inserts/s",
            (total_nodes + total_edges) as f64 / elapsed.as_secs_f64()
        );
    }
    println!("═══════════════════════════════════════════");

    Ok(())
}
