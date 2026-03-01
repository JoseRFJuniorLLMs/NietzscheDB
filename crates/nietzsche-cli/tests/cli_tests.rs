// ────────────────────────────────────────────────────────────────
//  nietzsche-cli  — unit tests (Sprint 3)
//
//  Strategy:
//    • The CLI is a binary crate (main.rs) with modules app.rs and ui.rs.
//      Since app/ui are private modules, we test through re-importing
//      the proto types and verifying the data structures that drive
//      the TUI display.
//    • TUI rendering tests use ratatui's TestBackend.
//    • Stress test and cluster binaries require a live server — we test
//      the pure-logic pieces (normalization, configs, proto types).
// ────────────────────────────────────────────────────────────────

use nietzsche_proto::db::{
    ConfigUpdate, CreateCollectionRequest, Empty, InsertRequest, MonitorRequest,
    SearchRequest, SystemStats,
};

// ═══════════════════════════════════════════════════════════════
// 1. SystemStats — proto struct
// ═══════════════════════════════════════════════════════════════

#[test]
fn system_stats_default() {
    let s = SystemStats::default();
    assert_eq!(s.total_collections, 0);
    assert_eq!(s.total_vectors, 0);
    assert!((s.total_memory_mb - 0.0).abs() < f64::EPSILON);
    assert!((s.qps - 0.0).abs() < f64::EPSILON);
}

#[test]
fn system_stats_custom_values() {
    let s = SystemStats {
        total_collections: 5,
        total_vectors: 100_000,
        total_memory_mb: 256.5,
        qps: 1500.75,
    };
    assert_eq!(s.total_collections, 5);
    assert_eq!(s.total_vectors, 100_000);
    assert!((s.total_memory_mb - 256.5).abs() < f64::EPSILON);
    assert!((s.qps - 1500.75).abs() < f64::EPSILON);
}

#[test]
fn system_stats_clone() {
    let s = SystemStats {
        total_collections: 3,
        total_vectors: 42,
        total_memory_mb: 10.0,
        qps: 99.9,
    };
    let s2 = s.clone();
    assert_eq!(s.total_collections, s2.total_collections);
    assert_eq!(s.total_vectors, s2.total_vectors);
}

#[test]
fn system_stats_debug() {
    let s = SystemStats::default();
    let dbg = format!("{s:?}");
    assert!(dbg.contains("total_collections"));
    assert!(dbg.contains("total_vectors"));
}

#[test]
fn system_stats_large_values() {
    let s = SystemStats {
        total_collections: 10_000,
        total_vectors: u64::MAX,
        total_memory_mb: 1_000_000.0,
        qps: 999_999.99,
    };
    assert_eq!(s.total_vectors, u64::MAX);
    assert!(s.total_memory_mb > 999_999.0);
}

// ═══════════════════════════════════════════════════════════════
// 2. CreateCollectionRequest — proto struct
// ═══════════════════════════════════════════════════════════════

#[test]
fn create_collection_request_fields() {
    let req = CreateCollectionRequest {
        name: "test_collection".to_string(),
        dimension: 128,
        metric: "l2".to_string(),
    };
    assert_eq!(req.name, "test_collection");
    assert_eq!(req.dimension, 128);
    assert_eq!(req.metric, "l2");
}

#[test]
fn create_collection_poincare() {
    let req = CreateCollectionRequest {
        name: "hyperbolic_64d".to_string(),
        dimension: 64,
        metric: "poincare".to_string(),
    };
    assert_eq!(req.metric, "poincare");
    assert_eq!(req.dimension, 64);
}

#[test]
fn create_collection_request_clone() {
    let req = CreateCollectionRequest {
        name: "test".to_string(),
        dimension: 8,
        metric: "poincare".to_string(),
    };
    let req2 = req.clone();
    assert_eq!(req.name, req2.name);
    assert_eq!(req.dimension, req2.dimension);
}

// ═══════════════════════════════════════════════════════════════
// 3. ConfigUpdate — proto struct
// ═══════════════════════════════════════════════════════════════

#[test]
fn config_update_ef_construction_only() {
    let cfg = ConfigUpdate {
        ef_construction: Some(50),
        ef_search: None,
        collection: "benchmark_8d".to_string(),
    };
    assert_eq!(cfg.ef_construction, Some(50));
    assert_eq!(cfg.ef_search, None);
    assert_eq!(cfg.collection, "benchmark_8d");
}

#[test]
fn config_update_ef_search_only() {
    let cfg = ConfigUpdate {
        ef_construction: None,
        ef_search: Some(100),
        collection: "col1".to_string(),
    };
    assert_eq!(cfg.ef_construction, None);
    assert_eq!(cfg.ef_search, Some(100));
}

#[test]
fn config_update_both() {
    let cfg = ConfigUpdate {
        ef_construction: Some(200),
        ef_search: Some(400),
        collection: "large".to_string(),
    };
    assert_eq!(cfg.ef_construction, Some(200));
    assert_eq!(cfg.ef_search, Some(400));
}

#[test]
fn config_update_default_collection() {
    let cfg = ConfigUpdate {
        ef_construction: None,
        ef_search: None,
        collection: String::new(),
    };
    assert!(cfg.collection.is_empty());
}

// ═══════════════════════════════════════════════════════════════
// 4. InsertRequest — stress test data structure
// ═══════════════════════════════════════════════════════════════

#[test]
fn insert_request_8d_poincare() {
    let req = InsertRequest {
        vector: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        id: 42,
        metadata: std::collections::HashMap::new(),
        typed_metadata: std::collections::HashMap::new(),
        collection: "benchmark_8d".to_string(),
        origin_node_id: String::new(),
        logical_clock: 0,
        durability: 0,
    };
    assert_eq!(req.vector.len(), 8);
    assert_eq!(req.id, 42);
    assert_eq!(req.collection, "benchmark_8d");
}

#[test]
fn insert_request_large_id() {
    let req = InsertRequest {
        vector: vec![0.0; 128],
        id: u32::MAX,
        metadata: std::collections::HashMap::new(),
        typed_metadata: std::collections::HashMap::new(),
        collection: "test".to_string(),
        origin_node_id: String::new(),
        logical_clock: 0,
        durability: 0,
    };
    assert_eq!(req.id, u32::MAX);
}

#[test]
fn insert_request_with_metadata() {
    let mut meta = std::collections::HashMap::new();
    meta.insert("category".to_string(), "test".to_string());
    meta.insert("source".to_string(), "benchmark".to_string());

    let req = InsertRequest {
        vector: vec![0.5; 8],
        id: 1,
        metadata: meta,
        typed_metadata: std::collections::HashMap::new(),
        collection: "col".to_string(),
        origin_node_id: String::new(),
        logical_clock: 0,
        durability: 0,
    };
    assert_eq!(req.metadata.len(), 2);
    assert_eq!(req.metadata.get("category").unwrap(), "test");
}

// ═══════════════════════════════════════════════════════════════
// 5. SearchRequest — stress test structure
// ═══════════════════════════════════════════════════════════════

#[test]
fn search_request_basic() {
    let req = SearchRequest {
        vector: vec![0.1; 8],
        top_k: 10,
        filter: std::collections::HashMap::new(),
        filters: Vec::new(),
        hybrid_query: None,
        hybrid_alpha: None,
        collection: "benchmark_8d".to_string(),
    };
    assert_eq!(req.top_k, 10);
    assert_eq!(req.vector.len(), 8);
}

#[test]
fn search_request_top_k_1() {
    let req = SearchRequest {
        vector: vec![0.0; 128],
        top_k: 1,
        filter: std::collections::HashMap::new(),
        filters: Vec::new(),
        hybrid_query: None,
        hybrid_alpha: None,
        collection: "test".to_string(),
    };
    assert_eq!(req.top_k, 1);
}

#[test]
fn search_request_clone() {
    let req = SearchRequest {
        vector: vec![0.5; 8],
        top_k: 5,
        filter: std::collections::HashMap::new(),
        filters: Vec::new(),
        hybrid_query: None,
        hybrid_alpha: None,
        collection: "c".to_string(),
    };
    let req2 = req.clone();
    assert_eq!(req.top_k, req2.top_k);
    assert_eq!(req.vector, req2.vector);
}

// ═══════════════════════════════════════════════════════════════
// 6. MonitorRequest / Empty — sentinel types
// ═══════════════════════════════════════════════════════════════

#[test]
fn monitor_request_default() {
    let _req = MonitorRequest {};
}

#[test]
fn empty_default() {
    let _e = Empty {};
    let e2 = Empty::default();
    let _ = format!("{e2:?}");
}

// ═══════════════════════════════════════════════════════════════
// 7. Poincaré ball normalization (stress test math)
//    Mirrors the normalization in stress.rs
// ═══════════════════════════════════════════════════════════════

fn stress_normalize(vector: &mut Vec<f64>) {
    let norm_sq: f64 = vector.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm >= 0.99 {
        let scale = 0.99 / norm;
        for x in vector.iter_mut() {
            *x *= scale;
        }
    }
}

#[test]
fn stress_normalize_inside_ball() {
    let mut v = vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
    let original = v.clone();
    stress_normalize(&mut v);
    assert_eq!(v, original); // norm ≈ 0.374 < 0.99
}

#[test]
fn stress_normalize_on_boundary() {
    // norm = sqrt(8 * 0.35^2) ≈ 0.99
    let mut v = vec![0.35; 8];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm >= 0.99 {
        stress_normalize(&mut v);
        let new_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(new_norm <= 0.99 + 1e-10);
    }
}

#[test]
fn stress_normalize_outside_ball() {
    let mut v = vec![0.5; 8]; // norm ≈ 1.414
    stress_normalize(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 0.99).abs() < 1e-10,
        "norm should be exactly 0.99, got {norm}"
    );
}

#[test]
fn stress_normalize_preserves_direction() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ratio_before = v[0] / v[7];
    stress_normalize(&mut v);
    let ratio_after = v[0] / v[7];
    assert!(
        (ratio_before - ratio_after).abs() < 1e-10,
        "direction should be preserved"
    );
}

#[test]
fn stress_normalize_zero_vector() {
    let mut v = vec![0.0; 8];
    stress_normalize(&mut v);
    assert_eq!(v, vec![0.0; 8]);
}

#[test]
fn stress_normalize_batch_1000() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let mut v: Vec<f64> = (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect();
        stress_normalize(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm <= 0.99 + 1e-10,
            "normalized vector has norm {norm} > 0.99"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// 8. TUI rendering — ratatui TestBackend
// ═══════════════════════════════════════════════════════════════

use ratatui::{
    backend::TestBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs, Wrap},
    Terminal,
};

/// Standalone reimplementation of the overview render
/// (since app/ui modules are private to the binary crate)
fn render_overview(stats: &SystemStats, area: Rect, terminal: &mut Terminal<TestBackend>) {
    terminal
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(1),
                ])
                .split(area);

            let stats_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[0]);

            let count_text = Paragraph::new(format!("{}", stats.total_vectors)).block(
                Block::default()
                    .title("Total Vectors")
                    .borders(Borders::ALL),
            );
            f.render_widget(count_text, stats_layout[0]);

            let cols_text = Paragraph::new(format!("{}", stats.total_collections))
                .block(Block::default().title("Collections").borders(Borders::ALL));
            f.render_widget(cols_text, stats_layout[1]);
        })
        .unwrap();
}

#[test]
fn tui_overview_renders_without_panic() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let stats = SystemStats {
        total_collections: 3,
        total_vectors: 1000,
        total_memory_mb: 64.5,
        qps: 500.0,
    };
    render_overview(&stats, Rect::new(0, 0, 80, 24), &mut terminal);
}

#[test]
fn tui_overview_renders_zero_stats() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let stats = SystemStats::default();
    render_overview(&stats, Rect::new(0, 0, 80, 24), &mut terminal);
}

#[test]
fn tui_overview_renders_large_numbers() {
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    let stats = SystemStats {
        total_collections: 10_000,
        total_vectors: 100_000_000,
        total_memory_mb: 32768.0,
        qps: 50000.0,
    };
    render_overview(&stats, Rect::new(0, 0, 120, 40), &mut terminal);
}

/// Render the full tab layout with tab selection
fn render_tabs(selected: usize, terminal: &mut Terminal<TestBackend>) {
    terminal
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(f.size());

            let titles = vec![
                "Overview [1]",
                "Collections [2]",
                "Storage [3]",
                "Admin [4]",
            ];
            let tabs = Tabs::new(titles)
                .select(selected)
                .block(
                    Block::default()
                        .title("NietzscheDB Mission Control")
                        .borders(Borders::ALL),
                )
                .highlight_style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                );
            f.render_widget(tabs, chunks[0]);

            let footer = Line::from(vec![
                Span::raw("Press "),
                Span::styled("Tab", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to switch tabs, "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to quit."),
            ]);
            f.render_widget(
                Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
                chunks[2],
            );
        })
        .unwrap();
}

#[test]
fn tui_tabs_renders_all_four() {
    for tab in 0..4 {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        render_tabs(tab, &mut terminal);
    }
}

/// Render collections list
fn render_collections(collections: &[String], terminal: &mut Terminal<TestBackend>) {
    terminal
        .draw(|f| {
            let items: Vec<Line> = collections
                .iter()
                .map(|c| Line::from(Span::raw(c)))
                .collect();
            let list = Paragraph::new(items).block(
                Block::default()
                    .title("Active Collections")
                    .borders(Borders::ALL),
            );
            f.render_widget(list, f.size());
        })
        .unwrap();
}

#[test]
fn tui_collections_empty() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    render_collections(&[], &mut terminal);
}

#[test]
fn tui_collections_multiple() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let cols = vec![
        "benchmark_8d".to_string(),
        "embeddings_1024".to_string(),
        "test_sync".to_string(),
    ];
    render_collections(&cols, &mut terminal);
}

/// Render admin/logs panel
fn render_admin(logs: &[String], terminal: &mut Terminal<TestBackend>) {
    terminal
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)])
                .split(f.size());

            let controls = "Actions: [S]napshot  [V]acuum";
            let p_controls = Paragraph::new(controls)
                .block(Block::default().title("Controls").borders(Borders::ALL));
            f.render_widget(p_controls, chunks[0]);

            let log_lines: Vec<Line> = logs
                .iter()
                .rev()
                .map(|s| Line::from(s.as_str()))
                .collect();
            let p_logs = Paragraph::new(log_lines)
                .block(Block::default().title("System Logs").borders(Borders::ALL))
                .wrap(Wrap { trim: true });
            f.render_widget(p_logs, chunks[1]);
        })
        .unwrap();
}

#[test]
fn tui_admin_empty_logs() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    render_admin(&[], &mut terminal);
}

#[test]
fn tui_admin_with_logs() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let logs = vec![
        "Ready. Waiting for connection...".to_string(),
        "Snapshot triggered...".to_string(),
        "Vacuum triggered...".to_string(),
    ];
    render_admin(&logs, &mut terminal);
}

/// Render storage panel
fn render_storage(stats: &SystemStats, terminal: &mut Terminal<TestBackend>) {
    terminal
        .draw(|f| {
            let info = format!(
                "Storage Mode: Multi-Collection\n\
                  Total Vectors: {}\n\
                  Total Memory: {:.2} MB\n\
                  (Detailed storage stats moved to Dashboard)",
                stats.total_vectors, stats.total_memory_mb
            );
            let p = Paragraph::new(info).block(
                Block::default()
                    .title("Storage Inspector")
                    .borders(Borders::ALL),
            );
            f.render_widget(p, f.size());
        })
        .unwrap();
}

#[test]
fn tui_storage_renders() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let stats = SystemStats {
        total_collections: 2,
        total_vectors: 50_000,
        total_memory_mb: 128.0,
        qps: 0.0,
    };
    render_storage(&stats, &mut terminal);
}

// ═══════════════════════════════════════════════════════════════
// 9. Tab cycling logic (mirrors CurrentTab::next)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Collections,
    Storage,
    Admin,
}

impl Tab {
    fn next(self) -> Self {
        match self {
            Tab::Overview => Tab::Collections,
            Tab::Collections => Tab::Storage,
            Tab::Storage => Tab::Admin,
            Tab::Admin => Tab::Overview,
        }
    }
}

#[test]
fn tab_cycle_forward() {
    let t = Tab::Overview;
    assert_eq!(t.next(), Tab::Collections);
    assert_eq!(t.next().next(), Tab::Storage);
    assert_eq!(t.next().next().next(), Tab::Admin);
    assert_eq!(t.next().next().next().next(), Tab::Overview); // full cycle
}

#[test]
fn tab_cycle_full_loop() {
    let start = Tab::Overview;
    let mut current = start;
    for _ in 0..4 {
        current = current.next();
    }
    assert_eq!(current, start);
}

#[test]
fn tab_as_usize_for_select() {
    assert_eq!(Tab::Overview as usize, 0);
    assert_eq!(Tab::Collections as usize, 1);
    assert_eq!(Tab::Storage as usize, 2);
    assert_eq!(Tab::Admin as usize, 3);
}

// ═══════════════════════════════════════════════════════════════
// 10. App state simulation (mirrors App struct)
// ═══════════════════════════════════════════════════════════════

struct AppSim {
    current_tab: Tab,
    should_quit: bool,
    stats: SystemStats,
    collections_list: Vec<String>,
    logs: Vec<String>,
}

impl AppSim {
    fn new() -> Self {
        Self {
            current_tab: Tab::Overview,
            should_quit: false,
            stats: SystemStats::default(),
            collections_list: Vec::new(),
            logs: vec!["Ready. Waiting for connection...".to_string()],
        }
    }
    fn next_tab(&mut self) {
        self.current_tab = self.current_tab.next();
    }
}

#[test]
fn app_initial_state() {
    let app = AppSim::new();
    assert_eq!(app.current_tab, Tab::Overview);
    assert!(!app.should_quit);
    assert_eq!(app.stats.total_vectors, 0);
    assert!(app.collections_list.is_empty());
    assert_eq!(app.logs.len(), 1);
    assert_eq!(app.logs[0], "Ready. Waiting for connection...");
}

#[test]
fn app_tab_cycling() {
    let mut app = AppSim::new();
    assert_eq!(app.current_tab, Tab::Overview);
    app.next_tab();
    assert_eq!(app.current_tab, Tab::Collections);
    app.next_tab();
    assert_eq!(app.current_tab, Tab::Storage);
    app.next_tab();
    assert_eq!(app.current_tab, Tab::Admin);
    app.next_tab();
    assert_eq!(app.current_tab, Tab::Overview);
}

#[test]
fn app_quit_flag() {
    let mut app = AppSim::new();
    assert!(!app.should_quit);
    app.should_quit = true;
    assert!(app.should_quit);
}

#[test]
fn app_stats_update() {
    let mut app = AppSim::new();
    app.stats = SystemStats {
        total_collections: 5,
        total_vectors: 10_000,
        total_memory_mb: 128.0,
        qps: 2500.0,
    };
    assert_eq!(app.stats.total_collections, 5);
    assert_eq!(app.stats.total_vectors, 10_000);
}

#[test]
fn app_collections_update() {
    let mut app = AppSim::new();
    app.collections_list = vec!["col_a".to_string(), "col_b".to_string()];
    assert_eq!(app.collections_list.len(), 2);
}

#[test]
fn app_log_append() {
    let mut app = AppSim::new();
    app.logs.push("Snapshot triggered...".to_string());
    app.logs.push("Vacuum triggered...".to_string());
    assert_eq!(app.logs.len(), 3);
    assert_eq!(app.logs.last().unwrap(), "Vacuum triggered...");
}

#[test]
fn app_direct_tab_select() {
    let mut app = AppSim::new();
    app.current_tab = Tab::Admin;
    assert_eq!(app.current_tab, Tab::Admin);
    app.current_tab = Tab::Storage;
    assert_eq!(app.current_tab, Tab::Storage);
}

// ═══════════════════════════════════════════════════════════════
// 11. Terminal backend sizing
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_backend_various_sizes() {
    // Ensure rendering doesn't panic at various terminal sizes
    for (w, h) in [(40, 10), (80, 24), (120, 40), (200, 60)] {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).unwrap();
        render_tabs(0, &mut terminal);
    }
}

#[test]
fn test_backend_minimum_size() {
    // Even very small terminals should render without panic
    let backend = TestBackend::new(20, 6);
    let mut terminal = Terminal::new(backend).unwrap();
    render_tabs(0, &mut terminal);
}
