pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NietzscheDB Dashboard</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {
    --bg: #0d0d0d; --panel: #161616; --border: #2a2a2a;
    --accent: #c9a84c; --accent2: #7b4ea0; --text: #e0e0e0;
    --muted: #666; --ok: #4caf50; --err: #e53935;
    --node-semantic: #c9a84c; --node-episodic: #4a90d9;
    --node-concept: #7b4ea0; --node-dream: #e67e22;
    --edge-color: #444; --font: 'Courier New', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }

  header {
    display: flex; align-items: center; gap: 16px;
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 { font-size: 18px; color: var(--accent); letter-spacing: 2px; }
  .stat-pill {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 10px; font-size: 12px;
  }
  .stat-pill span { color: var(--accent); font-weight: bold; }
  .health-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted); margin-left: auto;
  }
  .health-dot.ok { background: var(--ok); box-shadow: 0 0 6px var(--ok); }

  nav {
    display: flex; gap: 2px; padding: 8px 20px;
    background: var(--panel); border-bottom: 1px solid var(--border);
  }
  nav button {
    background: none; border: 1px solid var(--border); color: var(--muted);
    padding: 5px 16px; cursor: pointer; border-radius: 3px; font-family: var(--font);
    font-size: 12px; transition: all 0.15s;
  }
  nav button.active, nav button:hover {
    border-color: var(--accent); color: var(--accent); background: rgba(201,168,76,0.08);
  }

  .tab { display: none; height: calc(100vh - 92px); overflow: hidden; }
  .tab.active { display: flex; flex-direction: column; }

  /* ── GRAPH TAB ── */
  #tab-graph { position: relative; }
  #graph-svg { width: 100%; flex: 1; background: var(--bg); cursor: grab; }
  #graph-svg:active { cursor: grabbing; }
  .graph-toolbar {
    display: flex; gap: 8px; align-items: center;
    padding: 8px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .graph-toolbar button { padding: 4px 12px; }
  .node-tooltip {
    position: absolute; pointer-events: none; display: none;
    background: var(--panel); border: 1px solid var(--accent);
    border-radius: 4px; padding: 8px 12px; font-size: 12px;
    max-width: 280px; z-index: 10; line-height: 1.6;
  }
  .legend {
    position: absolute; bottom: 16px; left: 16px;
    background: rgba(22,22,22,0.9); border: 1px solid var(--border);
    border-radius: 4px; padding: 8px 12px;
  }
  .legend-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: 11px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

  /* ── PANEL LAYOUT (CRUD / NQL) ── */
  .panel-layout { display: flex; flex: 1; overflow: hidden; }
  .panel-left  { width: 340px; min-width: 340px; border-right: 1px solid var(--border); overflow-y: auto; padding: 16px; }
  .panel-right { flex: 1; overflow-y: auto; padding: 16px; }

  h2 { font-size: 13px; color: var(--accent); letter-spacing: 1px; margin-bottom: 12px; text-transform: uppercase; }
  h3 { font-size: 12px; color: var(--muted); margin: 16px 0 8px; text-transform: uppercase; }

  label { display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; margin-top: 10px; }
  input, select, textarea {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 6px 8px; border-radius: 3px;
    font-family: var(--font); font-size: 12px; outline: none;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 80px; }

  .btn {
    display: inline-block; padding: 6px 16px; border-radius: 3px; cursor: pointer;
    font-family: var(--font); font-size: 12px; border: 1px solid var(--accent);
    color: var(--accent); background: rgba(201,168,76,0.08); margin-top: 10px;
    transition: all 0.15s;
  }
  .btn:hover { background: rgba(201,168,76,0.2); }
  .btn-danger { border-color: var(--err); color: var(--err); background: rgba(229,57,53,0.08); }
  .btn-danger:hover { background: rgba(229,57,53,0.2); }
  .btn-purple { border-color: var(--accent2); color: var(--accent2); background: rgba(123,78,160,0.08); }
  .btn-purple:hover { background: rgba(123,78,160,0.2); }

  /* Result table */
  .result-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .result-table th {
    text-align: left; color: var(--muted); font-weight: normal;
    border-bottom: 1px solid var(--border); padding: 4px 8px;
  }
  .result-table td { padding: 5px 8px; border-bottom: 1px solid rgba(42,42,42,0.5); }
  .result-table tr:hover td { background: rgba(201,168,76,0.05); }
  .uuid { font-size: 10px; color: var(--muted); }
  .type-badge {
    font-size: 10px; padding: 1px 6px; border-radius: 2px;
    border: 1px solid; display: inline-block;
  }
  .type-Semantic      { border-color: var(--node-semantic); color: var(--node-semantic); }
  .type-Episodic      { border-color: var(--node-episodic); color: var(--node-episodic); }
  .type-Concept       { border-color: var(--node-concept);  color: var(--node-concept); }
  .type-DreamSnapshot { border-color: var(--node-dream);    color: var(--node-dream); }
  .type-Association      { border-color: #888; color: #888; }
  .type-Hierarchical     { border-color: var(--accent2); color: var(--accent2); }
  .type-LSystemGenerated { border-color: var(--ok); color: var(--ok); }
  .type-Pruned           { border-color: var(--err); color: var(--err); }

  /* NQL */
  #nql-input { font-size: 13px; min-height: 100px; line-height: 1.6; }
  .nql-result { margin-top: 16px; overflow-x: auto; }
  .msg { padding: 8px 12px; border-radius: 3px; font-size: 12px; margin: 8px 0; }
  .msg-ok  { background: rgba(76,175,80,0.1);  border: 1px solid var(--ok);  color: var(--ok); }
  .msg-err { background: rgba(229,57,53,0.1);  border: 1px solid var(--err); color: var(--err); }

  /* STATS */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; padding: 16px; }
  .stat-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px;
  }
  .stat-card .val { font-size: 32px; color: var(--accent); font-weight: bold; }
  .stat-card .lbl { color: var(--muted); font-size: 11px; margin-top: 4px; }
  .sleep-panel { padding: 16px; max-width: 500px; }
  .sleep-result { margin-top: 16px; }
  .sleep-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
  .sleep-row .val { color: var(--accent); }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <h1>⬡ NietzscheDB</h1>
  <div class="stat-pill">nodes: <span id="h-nodes">—</span></div>
  <div class="stat-pill">edges: <span id="h-edges">—</span></div>
  <div class="stat-pill">v<span id="h-ver">—</span></div>
  <div class="health-dot" id="health-dot" title="server health"></div>
</header>

<nav>
  <button class="active" onclick="switchTab('graph')">Graph</button>
  <button onclick="switchTab('nodes')">Nodes</button>
  <button onclick="switchTab('edges')">Edges</button>
  <button onclick="switchTab('nql')">NQL Console</button>
  <button onclick="switchTab('stats')">Stats</button>
</nav>

<!-- ── GRAPH ── -->
<div id="tab-graph" class="tab active">
  <div class="graph-toolbar">
    <button class="btn" onclick="loadGraph()">↺ Refresh</button>
    <button class="btn" onclick="resetZoom()">⊙ Reset Zoom</button>
    <span style="color:var(--muted);font-size:11px;margin-left:8px;">
      Drag to pan · Scroll to zoom · Click node for details
    </span>
  </div>
  <svg id="graph-svg"></svg>
  <div class="node-tooltip" id="tooltip"></div>
  <div class="legend">
    <div class="legend-row"><div class="legend-dot" style="background:var(--node-semantic)"></div>Semantic</div>
    <div class="legend-row"><div class="legend-dot" style="background:var(--node-episodic)"></div>Episodic</div>
    <div class="legend-row"><div class="legend-dot" style="background:var(--node-concept)"></div>Concept</div>
    <div class="legend-row"><div class="legend-dot" style="background:var(--node-dream)"></div>DreamSnapshot</div>
  </div>
</div>

<!-- ── NODES ── -->
<div id="tab-nodes" class="tab">
  <div class="panel-layout">
    <div class="panel-left">
      <h2>Insert Node</h2>
      <label>ID (leave empty for auto)</label>
      <input id="n-id" placeholder="uuid or empty">
      <label>Type</label>
      <select id="n-type">
        <option>Semantic</option><option>Episodic</option>
        <option>Concept</option><option>DreamSnapshot</option>
      </select>
      <label>Energy (0.0–1.0)</label>
      <input id="n-energy" type="number" min="0" max="1" step="0.01" value="1.0">
      <label>Content (JSON)</label>
      <textarea id="n-content" placeholder='{"label":"my concept"}'></textarea>
      <label>Embedding (comma-separated, e.g. 0.1,0.2,0.3,0.4)</label>
      <input id="n-embed" placeholder="0.1,0.2,0.3,0.4">
      <button class="btn" onclick="insertNode()">+ Insert Node</button>
      <div id="n-msg"></div>

      <h3>Get Node by ID</h3>
      <input id="n-get-id" placeholder="node uuid">
      <button class="btn" onclick="getNode()">Fetch</button>
    </div>
    <div class="panel-right">
      <h2>All Nodes <button class="btn" style="margin:0 0 0 8px;padding:3px 10px;font-size:11px" onclick="loadNodes()">↺</button></h2>
      <div id="nodes-table"></div>
    </div>
  </div>
</div>

<!-- ── EDGES ── -->
<div id="tab-edges" class="tab">
  <div class="panel-layout">
    <div class="panel-left">
      <h2>Insert Edge</h2>
      <label>From (node uuid)</label>
      <input id="e-from" placeholder="uuid">
      <label>To (node uuid)</label>
      <input id="e-to" placeholder="uuid">
      <label>Type</label>
      <select id="e-type">
        <option>Association</option><option>Hierarchical</option>
        <option>LSystemGenerated</option><option>Pruned</option>
      </select>
      <label>Weight</label>
      <input id="e-weight" type="number" min="0" step="0.01" value="1.0">
      <button class="btn" onclick="insertEdge()">+ Insert Edge</button>
      <div id="e-msg"></div>
    </div>
    <div class="panel-right">
      <h2>All Edges <button class="btn" style="margin:0 0 0 8px;padding:3px 10px;font-size:11px" onclick="loadEdges()">↺</button></h2>
      <div id="edges-table"></div>
    </div>
  </div>
</div>

<!-- ── NQL ── -->
<div id="tab-nql" class="tab">
  <div style="padding:16px;max-width:900px">
    <h2>NQL Console</h2>
    <label>Query</label>
    <textarea id="nql-input" placeholder="FIND NODES WHERE energy > 0.5 LIMIT 20"></textarea>
    <button class="btn" onclick="runNql()">▶ Run Query</button>
    <div id="nql-msg"></div>
    <div class="nql-result" id="nql-result"></div>
  </div>
</div>

<!-- ── STATS ── -->
<div id="tab-stats" class="tab">
  <div class="stats-grid">
    <div class="stat-card"><div class="val" id="s-nodes">—</div><div class="lbl">Total Nodes</div></div>
    <div class="stat-card"><div class="val" id="s-edges">—</div><div class="lbl">Total Edges</div></div>
    <div class="stat-card"><div class="val" id="s-ver">—</div><div class="lbl">Version</div></div>
    <div class="stat-card"><div class="val" id="s-health">—</div><div class="lbl">Health</div></div>
  </div>
  <div class="sleep-panel">
    <h2>Trigger Sleep Cycle</h2>
    <label>Noise</label><input id="sl-noise" type="number" value="0.02" step="0.01">
    <label>Adam Steps</label><input id="sl-steps" type="number" value="10">
    <label>Hausdorff Threshold</label><input id="sl-thresh" type="number" value="0.15" step="0.01">
    <button class="btn btn-purple" onclick="triggerSleep()">⟳ Run Sleep Cycle</button>
    <div id="sl-msg"></div>
    <div class="sleep-result" id="sl-result"></div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
const BASE = '';
let graphData = { nodes: [], edges: [] };
let svg, sim, g, zoom;

// ── Tabs ─────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'graph') loadGraph();
  if (name === 'nodes') loadNodes();
  if (name === 'edges') loadEdges();
  if (name === 'stats') loadStats();
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return r.json();
}

function msg(id, text, ok = true) {
  const el = document.getElementById(id);
  el.innerHTML = `<div class="msg ${ok ? 'msg-ok' : 'msg-err'}">${text}</div>`;
}

// ── Header stats ─────────────────────────────────────────────────────────────
async function refreshHeader() {
  try {
    const s = await api('/api/stats');
    document.getElementById('h-nodes').textContent = s.node_count;
    document.getElementById('h-edges').textContent = s.edge_count;
    document.getElementById('h-ver').textContent   = s.version;
    const h = await api('/api/health');
    const dot = document.getElementById('health-dot');
    dot.classList.toggle('ok', h.status === 'ok');
  } catch(e) { /* server may be loading */ }
}

// ── GRAPH ────────────────────────────────────────────────────────────────────
const typeColor = {
  Semantic:      '#c9a84c', Episodic: '#4a90d9',
  Concept:       '#7b4ea0', DreamSnapshot: '#e67e22',
};

function initSvg() {
  const el = document.getElementById('graph-svg');
  svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  zoom = d3.zoom().scaleExtent([0.05, 8]).on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoom);

  // arrow markers
  const defs = svg.append('defs');
  ['Association','Hierarchical','LSystemGenerated','Pruned'].forEach(t => {
    defs.append('marker')
      .attr('id', 'arrow-' + t)
      .attr('viewBox', '0 -5 10 10').attr('refX', 18).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', edgeColor(t));
  });

  g = svg.append('g');
}

function edgeColor(type) {
  return { Hierarchical:'#7b4ea0', LSystemGenerated:'#4caf50', Pruned:'#e53935' }[type] || '#555';
}

async function loadGraph() {
  initSvg();
  const data = await api('/api/graph');
  graphData = data;
  renderGraph(data);
  refreshHeader();
}

function renderGraph(data) {
  if (!g) initSvg();
  g.selectAll('*').remove();

  const el = document.getElementById('graph-svg');
  const W = el.clientWidth || 800, H = el.clientHeight || 600;
  const tooltip = document.getElementById('tooltip');

  // Build maps for link resolution
  const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
  const links = data.edges.map(e => ({
    source: e.from, target: e.to, type: e.edge_type, weight: e.weight, id: e.id,
  })).filter(l => nodeMap.has(l.source) && nodeMap.has(l.target));

  // Simulation
  sim = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(d => 80 / (d.weight || 1)))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide(18));

  // Edges
  const link = g.append('g').selectAll('line').data(links).join('line')
    .attr('stroke', d => edgeColor(d.type))
    .attr('stroke-width', d => Math.max(0.5, Math.min(3, d.weight)))
    .attr('stroke-opacity', 0.7)
    .attr('marker-end', d => `url(#arrow-${d.type})`);

  // Nodes
  const node = g.append('g').selectAll('circle').data(data.nodes).join('circle')
    .attr('r', d => 6 + d.energy * 8)
    .attr('fill', d => typeColor[d.node_type] || '#888')
    .attr('stroke', '#0d0d0d').attr('stroke-width', 1.5)
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on('mouseover', (e, d) => {
      const content = typeof d.content === 'object' ? JSON.stringify(d.content, null, 2) : d.content;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.offsetX + 12) + 'px';
      tooltip.style.top  = (e.offsetY + 12) + 'px';
      tooltip.innerHTML = `
        <b style="color:${typeColor[d.node_type]}">${d.node_type}</b><br>
        <span class="uuid">${d.id}</span><br>
        energy: <b>${d.energy.toFixed(3)}</b> | depth: <b>${d.depth.toFixed(3)}</b><br>
        ${content !== '{}' && content !== 'null' ? '<pre style="font-size:10px;margin-top:4px">' + content + '</pre>' : ''}
      `;
    })
    .on('mouseout', () => tooltip.style.display = 'none');

  // Labels (only when few nodes)
  if (data.nodes.length <= 80) {
    g.append('g').selectAll('text').data(data.nodes).join('text')
      .attr('font-size', 9).attr('fill', '#999').attr('dx', 10).attr('dy', 3)
      .text(d => {
        const label = d.content && d.content.label;
        return label || d.id.substring(0, 8) + '…';
      });
  }

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    if (data.nodes.length <= 80) {
      g.selectAll('text').attr('x', d => d.x).attr('y', d => d.y);
    }
  });
}

function resetZoom() {
  svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity);
}

// ── NODES ────────────────────────────────────────────────────────────────────
async function loadNodes() {
  const data = await api('/api/graph');
  const cont = document.getElementById('nodes-table');
  if (!data.nodes.length) { cont.innerHTML = '<p style="color:var(--muted);padding:8px">No nodes yet.</p>'; return; }
  cont.innerHTML = `
    <table class="result-table">
      <thead><tr><th>ID</th><th>Type</th><th>Energy</th><th>Content</th><th></th></tr></thead>
      <tbody>${data.nodes.map(n => `
        <tr>
          <td class="uuid">${n.id}</td>
          <td><span class="type-badge type-${n.node_type}">${n.node_type}</span></td>
          <td>${n.energy.toFixed(2)}</td>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
            ${n.content ? JSON.stringify(n.content) : ''}
          </td>
          <td><button class="btn btn-danger" style="padding:2px 8px;margin:0" onclick="delNode('${n.id}')">✕</button></td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

async function insertNode() {
  const embed = document.getElementById('n-embed').value.trim();
  const body = {
    id:        document.getElementById('n-id').value.trim() || undefined,
    node_type: document.getElementById('n-type').value,
    energy:    parseFloat(document.getElementById('n-energy').value),
    content:   (() => { try { return JSON.parse(document.getElementById('n-content').value || '{}'); } catch { return {}; } })(),
    embedding: embed ? embed.split(',').map(Number) : undefined,
  };
  const r = await api('/api/node', { method: 'POST', body: JSON.stringify(body) });
  if (r.id) { msg('n-msg', 'Created: ' + r.id); refreshHeader(); loadNodes(); }
  else       msg('n-msg', r.error || 'Error', false);
}

async function getNode() {
  const id = document.getElementById('n-get-id').value.trim();
  if (!id) return;
  const r = await api('/api/node/' + id);
  if (r.id) msg('n-msg', `<pre style="font-size:11px">${JSON.stringify(r, null, 2)}</pre>`);
  else       msg('n-msg', r.error || 'Not found', false);
}

async function delNode(id) {
  if (!confirm('Delete node ' + id + '?')) return;
  const r = await api('/api/node/' + id, { method: 'DELETE' });
  if (r.deleted) { refreshHeader(); loadNodes(); }
  else alert(r.error || 'Error');
}

// ── EDGES ────────────────────────────────────────────────────────────────────
async function loadEdges() {
  const data = await api('/api/graph');
  const cont = document.getElementById('edges-table');
  if (!data.edges.length) { cont.innerHTML = '<p style="color:var(--muted);padding:8px">No edges yet.</p>'; return; }
  cont.innerHTML = `
    <table class="result-table">
      <thead><tr><th>ID</th><th>From</th><th>To</th><th>Type</th><th>Weight</th><th></th></tr></thead>
      <tbody>${data.edges.map(e => `
        <tr>
          <td class="uuid">${e.id.substring(0,8)}…</td>
          <td class="uuid">${e.from.substring(0,8)}…</td>
          <td class="uuid">${e.to.substring(0,8)}…</td>
          <td><span class="type-badge type-${e.edge_type}">${e.edge_type}</span></td>
          <td>${e.weight.toFixed(2)}</td>
          <td><button class="btn btn-danger" style="padding:2px 8px;margin:0" onclick="delEdge('${e.id}')">✕</button></td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

async function insertEdge() {
  const body = {
    from:      document.getElementById('e-from').value.trim(),
    to:        document.getElementById('e-to').value.trim(),
    edge_type: document.getElementById('e-type').value,
    weight:    parseFloat(document.getElementById('e-weight').value),
  };
  if (!body.from || !body.to) { msg('e-msg', 'from and to are required', false); return; }
  const r = await api('/api/edge', { method: 'POST', body: JSON.stringify(body) });
  if (r.id) { msg('e-msg', 'Created: ' + r.id); refreshHeader(); loadEdges(); }
  else       msg('e-msg', r.error || 'Error', false);
}

async function delEdge(id) {
  if (!confirm('Delete edge ' + id + '?')) return;
  const r = await api('/api/edge/' + id, { method: 'DELETE' });
  if (r.deleted) { refreshHeader(); loadEdges(); }
  else alert(r.error || 'Error');
}

// ── NQL ──────────────────────────────────────────────────────────────────────
async function runNql() {
  const nql = document.getElementById('nql-input').value.trim();
  if (!nql) return;
  document.getElementById('nql-result').innerHTML = '';
  document.getElementById('nql-msg').innerHTML = '';
  const r = await api('/api/query', { method: 'POST', body: JSON.stringify({ nql }) });
  if (r.error) { msg('nql-msg', r.error, false); return; }
  const nodes = r.nodes || [];
  msg('nql-msg', `${nodes.length} node(s) returned`);
  if (!nodes.length) return;
  document.getElementById('nql-result').innerHTML = `
    <table class="result-table">
      <thead><tr><th>ID</th><th>Type</th><th>Energy</th><th>Content</th></tr></thead>
      <tbody>${nodes.map(n => `
        <tr>
          <td class="uuid">${n.id}</td>
          <td><span class="type-badge type-${n.node_type}">${n.node_type}</span></td>
          <td>${n.energy.toFixed(3)}</td>
          <td>${n.content ? JSON.stringify(n.content) : ''}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

// ── STATS ────────────────────────────────────────────────────────────────────
async function loadStats() {
  const [s, h] = await Promise.all([api('/api/stats'), api('/api/health')]);
  document.getElementById('s-nodes').textContent  = s.node_count;
  document.getElementById('s-edges').textContent  = s.edge_count;
  document.getElementById('s-ver').textContent    = s.version;
  document.getElementById('s-health').textContent = h.status === 'ok' ? '✓ OK' : '✗ ERROR';
  document.getElementById('s-health').style.color = h.status === 'ok' ? 'var(--ok)' : 'var(--err)';
}

async function triggerSleep() {
  const body = {
    noise:               parseFloat(document.getElementById('sl-noise').value),
    adam_steps:          parseInt(document.getElementById('sl-steps').value),
    hausdorff_threshold: parseFloat(document.getElementById('sl-thresh').value),
  };
  document.getElementById('sl-result').innerHTML = '<div class="msg">Running…</div>';
  const r = await api('/api/sleep', { method: 'POST', body: JSON.stringify(body) });
  if (r.error) { msg('sl-msg', r.error, false); document.getElementById('sl-result').innerHTML = ''; return; }
  document.getElementById('sl-result').innerHTML = `
    <div class="sleep-row"><span>Hausdorff before</span><span class="val">${r.hausdorff_before.toFixed(6)}</span></div>
    <div class="sleep-row"><span>Hausdorff after</span><span class="val">${r.hausdorff_after.toFixed(6)}</span></div>
    <div class="sleep-row"><span>Delta</span><span class="val">${r.hausdorff_delta.toFixed(6)}</span></div>
    <div class="sleep-row"><span>Committed</span><span class="val">${r.committed ? '✓ Yes' : '✗ No'}</span></div>
    <div class="sleep-row"><span>Nodes perturbed</span><span class="val">${r.nodes_perturbed}</span></div>`;
}

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => { if (graphData.nodes.length) renderGraph(graphData); });
initSvg();
loadGraph();
refreshHeader();
setInterval(refreshHeader, 10000);
</script>
</body>
</html>"#;
