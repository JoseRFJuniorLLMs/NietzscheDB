pub const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NietzscheDB</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c14;--panel:#161618;--panel2:#1e1e24;--border:#28283a;
  --text:#c8c8d8;--dim:#6a6a8a;--accent:#4a8aff;
  --cs:#c9a84c;--ce:#4a7fd4;--cc:#9b59b6;--cd:#e67e22;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* TOP BAR */
#topbar{height:42px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 12px;gap:10px;flex-shrink:0;z-index:200}
.tb-ws{font-size:14px;font-weight:500;cursor:pointer;padding:4px 8px;border-radius:4px;display:flex;align-items:center;gap:4px}
.tb-ws:hover{background:var(--panel2)}
.tb-sep{width:1px;height:20px;background:var(--border);flex-shrink:0}
.vbtn{background:none;border:1px solid transparent;color:var(--dim);padding:5px 12px;border-radius:4px;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:5px}
.vbtn.active,.vbtn:hover{background:var(--panel2);border-color:var(--border);color:var(--text)}
#tbr{margin-left:auto;display:flex;align-items:center;gap:6px}
.tbi{background:none;border:none;color:var(--dim);cursor:pointer;padding:5px 7px;border-radius:4px;font-size:14px}
.tbi:hover{background:var(--panel2);color:var(--text)}

/* MAIN */
#main{display:flex;flex:1;overflow:hidden}

/* LEFT PANEL */
#lp{width:238px;min-width:238px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden}
#gi{padding:14px;border-bottom:1px solid var(--border)}
#gtitle{font-size:14px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between}
.gi-row{display:flex;gap:24px;margin-bottom:6px}
.gi-block .gi-l{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em}
.gi-block .gi-v{font-size:22px;font-weight:700;line-height:1.1}
.gi-t{font-size:11px;color:var(--dim);margin-top:4px}
#sw{padding:8px 10px;border-bottom:1px solid var(--border);display:flex;gap:4px}
#si{flex:1;background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:4px;font-size:12px}
#si:focus{outline:none;border-color:var(--accent)}
.sbtn{background:none;border:1px solid var(--border);color:var(--dim);padding:5px 8px;border-radius:4px;cursor:pointer;font-size:13px}
.sbtn:hover{color:var(--text)}
#sr{padding:0;max-height:120px;overflow-y:auto;border-bottom:1px solid var(--border)}
.sri{display:flex;align-items:center;gap:6px;padding:5px 10px;cursor:pointer;font-size:11px}
.sri:hover{background:var(--panel2)}

/* ACCORDION */
.ac{border-bottom:1px solid var(--border)}
.ac-h{display:flex;align-items:center;justify-content:space-between;padding:9px 14px;cursor:pointer;font-size:12px;color:var(--dim);user-select:none}
.ac-h:hover,.ac-h.open{color:var(--text);background:var(--panel2)}
.ac-arr{font-size:10px;transition:transform .2s}
.ac-h.open .ac-arr{transform:rotate(90deg)}
.ac-b{display:none;padding:10px 14px}
.ac-h.open + .ac-b{display:block}
.al{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;margin-top:8px}
.al:first-child{margin-top:0}
.asel{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:4px;font-size:12px}
.asel:focus{outline:none;border-color:var(--accent)}
.atr{display:flex;align-items:center;justify-content:space-between;margin-top:8px;font-size:12px}
.tsw{position:relative;width:32px;height:16px;display:inline-block;flex-shrink:0}
.tsw input{opacity:0;width:0;height:0}
.tsw .k{position:absolute;inset:0;background:#333;border-radius:8px;cursor:pointer;transition:.2s}
.tsw input:checked + .k{background:var(--accent)}
.tsw .k:before{content:'';position:absolute;width:12px;height:12px;background:#fff;border-radius:50%;left:2px;top:2px;transition:.2s}
.tsw input:checked + .k:before{left:18px}

/* LAYOUT BTN */
.la-btns{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px}
.la-b{background:var(--panel2);border:1px solid var(--border);color:var(--dim);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px}
.la-b.active,.la-b:hover{border-color:var(--accent);color:var(--accent)}
.la-row{display:flex;gap:6px}
.run-b{flex:1;background:var(--accent);color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:500}
.run-b:hover{opacity:.9}
.stp-b{background:var(--panel2);border:1px solid var(--border);color:var(--dim);padding:6px 10px;border-radius:4px;cursor:pointer;font-size:12px}
.stp-b:hover{color:var(--text)}

/* FILTER */
.fi{display:flex;align-items:center;gap:6px;padding:4px 0;cursor:pointer;font-size:12px}
.fi:hover{color:#fff}
.fd{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.fl{flex:1}
.fc{font-size:11px;color:var(--dim)}
input[type=range]{width:100%;accent-color:var(--accent);height:14px;margin:2px 0}
.fr-lbl{display:flex;justify-content:space-between;font-size:10px;color:var(--dim);margin-bottom:1px}

/* METRICS */
.mb{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:6px;border-radius:4px;cursor:pointer;font-size:11px;text-align:left;margin-bottom:4px}
.mb:hover{border-color:var(--accent);color:var(--accent)}
.mr{font-size:11px;color:var(--dim);margin-top:6px;padding:6px;background:var(--bg);border-radius:4px;line-height:1.6}

/* CANVAS WRAP */
#cw{position:relative;flex:1;overflow:hidden;background:var(--bg)}
#gc{position:absolute;top:0;left:0;width:100%;height:100%}

/* RIGHT TOOLBAR */
#rt{position:absolute;right:10px;top:50%;transform:translateY(-50%);display:flex;flex-direction:column;gap:2px;z-index:20}
.rtb{width:32px;height:32px;background:rgba(22,22,24,.92);border:1px solid var(--border);color:var(--dim);border-radius:4px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:15px;backdrop-filter:blur(4px)}
.rtb:hover{color:var(--text);border-color:#444466}
.rtb.active{color:var(--accent);border-color:var(--accent);background:rgba(74,138,255,.12)}
.rt-sep{height:6px}

/* BOTTOM BAR */
#bb{position:absolute;bottom:0;left:0;right:0;background:rgba(18,18,26,.94);border-top:1px solid var(--border);display:flex;align-items:stretch;backdrop-filter:blur(6px);z-index:20;min-height:88px}
.bbs{padding:10px 14px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:3px;min-width:130px}
.bbt{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em}
.bbv{font-size:12px;font-weight:500;margin-bottom:3px}
.bbl{display:flex;flex-direction:column;gap:3px;max-height:58px;overflow-y:auto}
.bbli{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--dim);cursor:pointer}
.bbli:hover{color:var(--text)}
.bbd{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.bbsz{display:flex;align-items:center;gap:6px;margin-top:2px}
.bbc{border-radius:50%;background:var(--dim);flex-shrink:0}

/* LOADING */
#ld{position:absolute;inset:0;background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;z-index:300}
.sp{width:28px;height:28px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* TOOLTIP */
#tt{position:absolute;background:rgba(20,20,28,.97);border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:11px;pointer-events:none;z-index:200;max-width:210px;display:none;backdrop-filter:blur(4px)}
#tt.show{display:block}
.tth{font-weight:600;margin-bottom:5px;font-size:12px}
.ttr{display:flex;justify-content:space-between;gap:12px;color:var(--dim);margin-top:2px}
.ttv{color:var(--text)}

/* NQL */
#nqlp{display:none;position:absolute;top:44px;right:52px;width:400px;background:rgba(20,20,28,.97);border:1px solid var(--border);border-radius:8px;padding:14px;z-index:100;backdrop-filter:blur(8px)}
#nqlp.open{display:block}
#nqlh{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:13px;font-weight:500}
#nqli{width:100%;background:var(--bg);border:1px solid var(--border);color:#c9a84c;font-family:monospace;font-size:12px;padding:8px;border-radius:4px;resize:vertical;min-height:72px}
.nqlrow{display:flex;gap:6px;margin-top:8px}
.nqlrun{flex:1;background:var(--accent);color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:500}
.nqlclr{background:var(--panel2);border:1px solid var(--border);color:var(--dim);padding:6px 10px;border-radius:4px;cursor:pointer;font-size:12px}
#nqlr{margin-top:8px;max-height:160px;overflow-y:auto;font-size:11px;color:var(--dim);line-height:1.6}
#nqlfab{position:absolute;bottom:96px;left:10px;background:rgba(22,22,24,.9);border:1px solid var(--border);color:var(--dim);padding:5px 10px;border-radius:4px;cursor:pointer;font-size:11px;font-family:monospace;z-index:20;backdrop-filter:blur(4px)}
#nqlfab:hover{color:#c9a84c;border-color:#c9a84c50}

/* DATA VIEW */
#dv{flex:1;display:none;flex-direction:column;overflow:hidden}
#dtb{display:flex;padding:8px 14px;gap:6px;border-bottom:1px solid var(--border);background:var(--panel)}
.dtbb{background:none;border:1px solid var(--border);color:var(--dim);padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px}
.dtbb.active{border-color:var(--accent);color:var(--accent)}
#dtw{flex:1;overflow:auto;padding:10px}
#dtbl{width:100%;border-collapse:collapse;font-size:12px}
#dtbl th{background:var(--panel);padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);color:var(--dim);font-weight:500;position:sticky;top:0;z-index:10}
#dtbl td{padding:5px 10px;border-bottom:1px solid #1e1e2a;color:var(--text)}
#dtbl tr:hover td{background:var(--panel2)}

/* AGENCY VIEW */
.ag-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px;width:calc(50% - 8px);min-width:300px;display:inline-block;vertical-align:top;margin:0 8px 16px 0}
.ag-title{font-size:14px;font-weight:600;margin-bottom:12px;color:var(--accent);display:flex;align-items:center;gap:6px}
.ag-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.ag-metric{background:var(--panel2);border-radius:4px;padding:8px 10px;display:flex;flex-direction:column;gap:2px}
.ag-label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.ag-val{font-size:18px;font-weight:700;line-height:1.2}
.ag-val.good{color:#2ecc71}.ag-val.warn{color:#e67e22}.ag-val.bad{color:#e74c3c}
.ag-list{max-height:200px;overflow-y:auto}
.ag-item{background:var(--panel2);border-radius:4px;padding:8px 10px;margin-bottom:4px;font-size:12px;display:flex;justify-content:space-between;align-items:center}
.ag-item .ag-pri{font-weight:600;color:var(--accent)}
.ag-bar{height:4px;border-radius:2px;background:var(--border);margin-top:4px;overflow:hidden}
.ag-bar-fill{height:100%;border-radius:2px;background:var(--accent);transition:width .3s}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div class="tb-ws">Workspace <span style="font-size:10px">▾</span></div>
  <div class="tb-sep"></div>
  <button class="vbtn active" id="vg">⊞ Graph</button>
  <button class="vbtn" id="vd">≡ Data</button>
  <button class="vbtn" id="va">◎ Agency</button>
  <div id="tbr">
    <button class="tbi" id="btnNql" title="NQL Console">⌨</button>
    <button class="tbi" title="Refresh" onclick="loadGraph()">↺</button>
    <div class="tb-sep"></div>
    <span style="font-size:12px;color:var(--dim)">NietzscheDB</span>
  </div>
</div>

<!-- MAIN -->
<div id="main">

  <!-- LEFT PANEL -->
  <div id="lp">
    <!-- Graph info -->
    <div id="gi">
      <div id="gtitle">NietzscheDB <button class="tbi" style="font-size:11px">✎</button></div>
      <div class="gi-row">
        <div class="gi-block"><div class="gi-l">Nodes</div><div class="gi-v" id="nc">0</div></div>
        <div class="gi-block"><div class="gi-l">Edges</div><div class="gi-v" id="ec">0</div></div>
      </div>
      <div class="gi-t">Hyperbolic graph database</div>
    </div>

    <!-- Search -->
    <div id="sw">
      <input type="text" id="si" placeholder="Search nodes, edges">
      <button class="sbtn" onclick="doSearch()">⌕</button>
    </div>
    <div id="sr" style="display:none"></div>

    <!-- Layout -->
    <div class="ac">
      <div class="ac-h open" onclick="this.classList.toggle('open')">
        <span>⊞ Layout</span><span class="ac-arr">›</span>
      </div>
      <div class="ac-b">
        <div class="la-btns" id="algos">
          <button class="la-b active" data-a="force">Force</button>
          <button class="la-b" data-a="fa2">ForceAtlas2</button>
          <button class="la-b" data-a="circular">Circular</button>
          <button class="la-b" data-a="random">Random</button>
        </div>
        <div class="la-row">
          <button class="run-b" id="btnRun">▶ Run</button>
          <button class="stp-b" id="btnStop">⏹</button>
        </div>
      </div>
    </div>

    <!-- Appearance -->
    <div class="ac">
      <div class="ac-h open" onclick="this.classList.toggle('open')">
        <span>◎ Appearance</span><span class="ac-arr">›</span>
      </div>
      <div class="ac-b">
        <div class="al">Color nodes by</div>
        <select class="asel" id="colorBy">
          <option value="node_type">node_type</option>
          <option value="energy">energy</option>
          <option value="depth">depth</option>
          <option value="modularity">modularity class</option>
        </select>
        <div class="al">Size nodes by</div>
        <select class="asel" id="sizeBy">
          <option value="uniform">Uniform</option>
          <option value="energy">energy</option>
          <option value="degree">Degree (dynamic)</option>
        </select>
        <div class="al">Size edges by</div>
        <select class="asel" id="sizeEdge">
          <option value="weight">weight</option>
          <option value="uniform">Uniform</option>
        </select>
        <div class="atr">Labels <label class="tsw"><input type="checkbox" id="chkLabels" checked><span class="k"></span></label></div>
        <div class="atr">Glow mode <label class="tsw"><input type="checkbox" id="chkGlow" checked><span class="k"></span></label></div>
        <div class="atr">Show edges <label class="tsw"><input type="checkbox" id="chkEdges" checked><span class="k"></span></label></div>
      </div>
    </div>

    <!-- Filters -->
    <div class="ac">
      <div class="ac-h open" onclick="this.classList.toggle('open')">
        <span>⊟ Filters</span><span class="ac-arr">›</span>
      </div>
      <div class="ac-b">
        <div id="ftypes"></div>
        <div class="al" style="margin-top:10px">energy</div>
        <div class="fr-lbl"><span id="felol">0.00</span><span id="fehil">1.00</span></div>
        <input type="range" id="felo" min="0" max="100" value="0">
        <input type="range" id="fehi" min="0" max="100" value="100">
        <div class="al">depth</div>
        <div class="fr-lbl"><span id="fdlol">0.00</span><span id="fdhil">1.00</span></div>
        <input type="range" id="fdlo" min="0" max="100" value="0">
        <input type="range" id="fdhi" min="0" max="100" value="100">
      </div>
    </div>

    <!-- Metrics -->
    <div class="ac">
      <div class="ac-h" onclick="this.classList.toggle('open')">
        <span>⊕ Metrics</span><span class="ac-arr">›</span>
      </div>
      <div class="ac-b">
        <button class="mb" onclick="runM('degree')">Degree Centrality</button>
        <button class="mb" onclick="runM('weighted')">Weighted Degree</button>
        <button class="mb" onclick="runM('modularity')">Modularity Class</button>
        <button class="mb" onclick="runM('density')">Graph Density</button>
        <div id="mr" class="mr" style="display:none"></div>
      </div>
    </div>
  </div>

  <!-- GRAPH VIEW -->
  <div id="cw">
    <div id="ld"><div class="sp"></div><div style="color:var(--dim)">Loading graph…</div></div>
    <canvas id="gc"></canvas>

    <!-- Right toolbar -->
    <div id="rt">
      <button class="rtb active" id="rtSel" title="Select">↖</button>
      <button class="rtb" id="rtBox" title="Box select">⬚</button>
      <div class="rt-sep"></div>
      <button class="rtb" id="rtZin" title="Zoom in">+</button>
      <button class="rtb" id="rtZout" title="Zoom out">−</button>
      <button class="rtb" id="rtZrect" title="Zoom rect">⊡</button>
      <button class="rtb" id="rtFit" title="Fit view">⊞</button>
    </div>

    <!-- Bottom bar -->
    <div id="bb">
      <div class="bbs">
        <div class="bbt">Color nodes by</div>
        <div class="bbv" id="bbc">node_type</div>
        <div class="bbl" id="bbcl"></div>
      </div>
      <div class="bbs">
        <div class="bbt">Size nodes by</div>
        <div class="bbv" id="bbs2">Uniform</div>
        <div class="bbsz">
          <div class="bbc" style="width:5px;height:5px"></div>
          <div class="bbc" style="width:10px;height:10px"></div>
          <div class="bbc" style="width:16px;height:16px"></div>
          <div class="bbc" style="width:22px;height:22px"></div>
          <span style="font-size:10px;color:var(--dim);margin-left:4px" id="bbs2r">min — max</span>
        </div>
      </div>
      <div class="bbs">
        <div class="bbt">Size edges by</div>
        <div class="bbv" id="bbe">weight</div>
        <div class="bbsz">
          <div style="height:1px;width:18px;background:var(--dim)"></div>
          <div style="height:2px;width:18px;background:var(--dim)"></div>
          <div style="height:4px;width:18px;background:var(--dim)"></div>
          <span style="font-size:10px;color:var(--dim);margin-left:4px" id="bber">1 — 1</span>
        </div>
      </div>
    </div>

    <!-- NQL FAB -->
    <button id="nqlfab">⌨ NQL</button>

    <!-- Tooltip -->
    <div id="tt"></div>

    <!-- NQL overlay -->
    <div id="nqlp">
      <div id="nqlh">
        NQL Console
        <button class="tbi" id="nqlClose">✕</button>
      </div>
      <textarea id="nqli" placeholder="MATCH (n:Semantic) WHERE n.energy > 0.5 RETURN n LIMIT 20"></textarea>
      <div class="nqlrow">
        <button class="nqlrun" id="nqlRun">▶ Run</button>
        <button class="nqlclr" id="nqlClr">Clear</button>
      </div>
      <div id="nqlr"></div>
    </div>
  </div>

  <!-- DATA VIEW -->
  <div id="dv">
    <div id="dtb">
      <button class="dtbb active" id="dtNodes">Nodes</button>
      <button class="dtbb" id="dtEdges">Edges</button>
    </div>
    <div id="dtw">
      <table id="dtbl">
        <thead id="dth"></thead>
        <tbody id="dtbd"></tbody>
      </table>
    </div>
  </div>

  <!-- AGENCY VIEW -->
  <div id="av" style="display:none;flex:1;padding:20px;overflow-y:auto;gap:16px;flex-wrap:wrap;align-content:flex-start">
    <!-- Health Card -->
    <div class="ag-card" id="ag-health">
      <div class="ag-title">Health Report</div>
      <div class="ag-grid" id="ag-health-grid">
        <div class="ag-metric"><span class="ag-label">Mean Energy</span><span class="ag-val" id="ag-energy">—</span></div>
        <div class="ag-metric"><span class="ag-label">Hausdorff</span><span class="ag-val" id="ag-hausdorff">—</span></div>
        <div class="ag-metric"><span class="ag-label">Coherence</span><span class="ag-val" id="ag-coherence">—</span></div>
        <div class="ag-metric"><span class="ag-label">Gaps</span><span class="ag-val" id="ag-gaps">—</span></div>
        <div class="ag-metric"><span class="ag-label">Entropy Spikes</span><span class="ag-val" id="ag-entropy">—</span></div>
        <div class="ag-metric"><span class="ag-label">Fractal</span><span class="ag-val" id="ag-fractal">—</span></div>
        <div class="ag-metric"><span class="ag-label">Nodes</span><span class="ag-val" id="ag-nodes">—</span></div>
        <div class="ag-metric"><span class="ag-label">Edges</span><span class="ag-val" id="ag-edges">—</span></div>
      </div>
    </div>
    <!-- Observer Identity -->
    <div class="ag-card" id="ag-observer">
      <div class="ag-title">Observer Identity</div>
      <div class="ag-grid" id="ag-obs-grid">
        <div class="ag-metric"><span class="ag-label">Energy</span><span class="ag-val" id="ag-obs-energy">—</span></div>
        <div class="ag-metric"><span class="ag-label">Depth</span><span class="ag-val" id="ag-obs-depth">—</span></div>
        <div class="ag-metric"><span class="ag-label">Hausdorff</span><span class="ag-val" id="ag-obs-hausdorff">—</span></div>
      </div>
    </div>
    <!-- Desires -->
    <div class="ag-card" id="ag-desires">
      <div class="ag-title">Motor de Desejo</div>
      <div id="ag-desire-list" class="ag-list"></div>
    </div>
    <!-- Evolution -->
    <div class="ag-card" id="ag-evolution">
      <div class="ag-title">Rule Evolution</div>
      <div class="ag-grid">
        <div class="ag-metric"><span class="ag-label">Generation</span><span class="ag-val" id="ag-evo-gen">—</span></div>
        <div class="ag-metric"><span class="ag-label">Strategy</span><span class="ag-val" id="ag-evo-strat">—</span></div>
      </div>
      <div id="ag-evo-history" class="ag-list" style="margin-top:8px"></div>
    </div>
    <!-- Narrative -->
    <div class="ag-card" id="ag-narrative">
      <div class="ag-title">Narrative</div>
      <div id="ag-narr-text" style="font-style:italic;color:var(--dim);line-height:1.5"></div>
    </div>
  </div>

</div><!-- /main -->

<script>
// ── Constants ──────────────────────────────────────────────────────────
const TC = {Semantic:'#c9a84c',Episodic:'#4a7fd4',Concept:'#9b59b6',DreamSnapshot:'#e67e22'};
const MC = ['#4a7fd4','#e74c3c','#2ecc71','#9b59b6','#e67e22','#c9a84c','#1abc9c','#e91e63','#3498db','#f39c12'];

function h2r(h){return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]}
function v2c(v){
  const t=Math.max(0,Math.min(1,v));
  return `rgb(${Math.round(t<.5?t*2*255:255)},${Math.round(t<.5?t*2*180:(1-t)*2*180)},${Math.round(t<.5?255:0)})`;
}

// ── State ────────────────────────────────────────────────────────────
let AN=[],AE=[],VN=[],SL=[];
let deg=new Map(), wdeg=new Map();
let sim=null, algo='force';
let xf={x:0,y:0,k:1};
let panning=false,ps=null,dragging=false,dn=null;
let hl=new Set();
let F={types:new Set(Object.keys(TC)),elo:0,ehi:1,dlo:0,dhi:1};
let AP={colorBy:'node_type',sizeBy:'uniform',sizeEdge:'weight',labels:true,glow:true,edges:true};

const cv=document.getElementById('gc');
const cx=cv.getContext('2d');

function rsz(){cv.width=cv.clientWidth;cv.height=cv.clientHeight}
window.addEventListener('resize',()=>{rsz();draw()});
rsz();

// ── Helpers ───────────────────────────────────────────────────────────
function nColor(n){
  if(AP.colorBy==='node_type') return TC[n.node_type]||'#888';
  if(AP.colorBy==='energy') return v2c(n.energy||0);
  if(AP.colorBy==='depth') return v2c(n.depth||0);
  if(AP.colorBy==='modularity') return MC[(n.mc||0)%MC.length];
  return '#888';
}
function nRad(n){
  if(AP.sizeBy==='energy') return 5+(n.energy||0)*20;
  if(AP.sizeBy==='degree'){const d=deg.get(n.id)||0,mx=Math.max(...deg.values())||1;return 5+(d/mx)*20}
  return 10;
}
function eW(e){return AP.sizeEdge==='weight'?Math.max(0.5,(e.weight||1)*1.2):1}
function s2g(x,y){return{x:(x-xf.x)/xf.k,y:(y-xf.y)/xf.k}}
function nAt(sx,sy){
  const g=s2g(sx,sy);
  for(let i=VN.length-1;i>=0;i--){
    const n=VN[i];if(n.x==null)continue;
    const r=nRad(n)+6;if((n.x-g.x)**2+(n.y-g.y)**2<r*r)return n;
  }return null;
}
function nLabel(n){
  if(n.label) return n.label;
  const c=n.content;
  if(c&&typeof c==='object'){if(c.label)return c.label;if(c.name)return c.name;if(c.title)return c.title}
  return n.id.slice(0,8);
}

// ── Draw ──────────────────────────────────────────────────────────────
function draw(){
  const W=cv.width,H=cv.height;
  cx.fillStyle='#0c0c14';cx.fillRect(0,0,W,H);
  cx.save();cx.translate(xf.x,xf.y);cx.scale(xf.k,xf.k);

  // Edges
  if(AP.edges){
    cx.globalCompositeOperation='source-over';
    SL.forEach(lk=>{
      const s=lk.source,t=lk.target;
      if(!s||!t||s.x==null||t.x==null)return;
      const [r,g,b]=h2r(nColor(s));
      cx.beginPath();cx.moveTo(s.x,s.y);cx.lineTo(t.x,t.y);
      cx.lineWidth=Math.max(0.4/xf.k,eW(lk)/xf.k);
      cx.strokeStyle=`rgba(${r},${g},${b},0.22)`;cx.stroke();
    });
  }

  // Glow pass
  if(AP.glow){
    cx.globalCompositeOperation='lighter';
    VN.forEach(n=>{
      if(n.x==null)return;
      const [r,g,b]=h2r(nColor(n));
      const rd=nRad(n);
      const gr=cx.createRadialGradient(n.x,n.y,0,n.x,n.y,rd*5);
      const a=(hl.size===0||hl.has(n.id))?0.3:0.04;
      gr.addColorStop(0,`rgba(${r},${g},${b},${a})`);
      gr.addColorStop(1,'rgba(0,0,0,0)');
      cx.fillStyle=gr;cx.beginPath();cx.arc(n.x,n.y,rd*5,0,Math.PI*2);cx.fill();
    });
  }

  // Node circles
  cx.globalCompositeOperation='source-over';
  VN.forEach(n=>{
    if(n.x==null)return;
    const [r,g,b]=h2r(nColor(n));
    const rd=nRad(n);
    const dim=hl.size>0&&!hl.has(n.id);
    cx.beginPath();cx.arc(n.x,n.y,rd,0,Math.PI*2);
    cx.fillStyle=`rgba(${r},${g},${b},${dim?0.2:0.9})`;cx.fill();
    cx.strokeStyle=`rgba(${Math.min(r+50,255)},${Math.min(g+50,255)},${Math.min(b+50,255)},${dim?0.1:0.55})`;
    cx.lineWidth=1.5;cx.stroke();
  });

  // Labels
  if(AP.labels){
    cx.globalCompositeOperation='source-over';
    VN.forEach(n=>{
      if(n.x==null)return;
      const rd=nRad(n);
      if(xf.k*rd<3)return;
      const fs=Math.max(9,Math.min(13,rd*0.85));
      const lbl=nLabel(n);
      const dim=hl.size>0&&!hl.has(n.id);
      cx.font=`${fs}px -apple-system,sans-serif`;
      cx.textAlign='center';cx.textBaseline='middle';
      // shadow
      cx.fillStyle=`rgba(0,0,0,${dim?0.2:0.7})`;
      cx.fillText(lbl,n.x+1,n.y+1);
      const [r,g,b]=h2r(nColor(n));
      cx.fillStyle=dim?`rgba(${r},${g},${b},0.25)`:'rgba(255,255,255,0.92)';
      cx.fillText(lbl,n.x,n.y);
    });
  }
  cx.restore();
}

// ── Simulation ────────────────────────────────────────────────────────
function buildSL(){
  const m=new Map(VN.map(n=>[n.id,n]));
  SL=AE.filter(e=>m.has(e.from)&&m.has(e.to))
    .map(e=>({source:e.from,target:e.to,weight:e.weight||1,edge_type:e.edge_type||''}));
}
function compDeg(){
  deg.clear();wdeg.clear();
  VN.forEach(n=>{deg.set(n.id,0);wdeg.set(n.id,0)});
  SL.forEach(lk=>{
    const si=typeof lk.source==='object'?lk.source.id:lk.source;
    const ti=typeof lk.target==='object'?lk.target.id:lk.target;
    deg.set(si,(deg.get(si)||0)+1);deg.set(ti,(deg.get(ti)||0)+1);
    wdeg.set(si,(wdeg.get(si)||0)+(lk.weight||1));wdeg.set(ti,(wdeg.get(ti)||0)+(lk.weight||1));
  });
}
function initSim(){
  const W=cv.width,H=cv.height;
  buildSL();compDeg();
  if(sim)sim.stop();
  sim=d3.forceSimulation(VN)
    .force('link',d3.forceLink(SL).id(d=>d.id).distance(65).strength(0.4))
    .force('charge',d3.forceManyBody().strength(-130))
    .force('center',d3.forceCenter(W/2,H/2))
    .force('col',d3.forceCollide().radius(n=>nRad(n)+5))
    .alphaMin(0.001).on('tick',draw);
}
function applyLayout(){
  const W=cv.width,H=cv.height;
  if(algo==='circular'){
    const byT={};VN.forEach(n=>{(byT[n.node_type]||(byT[n.node_type]=[])).push(n)});
    const types=Object.keys(byT);
    types.forEach((t,ti)=>{
      const grp=byT[t];
      const ga=ti/types.length*Math.PI*2;
      const cx2=W/2+Math.cos(ga)*Math.min(W,H)*.27;
      const cy2=H/2+Math.sin(ga)*Math.min(W,H)*.27;
      grp.forEach((n,i)=>{
        const a=i/grp.length*Math.PI*2;
        n.x=cx2+Math.cos(a)*Math.min(W,H)*.1;n.y=cy2+Math.sin(a)*Math.min(W,H)*.1;n.vx=n.vy=0;
      });
    });
    if(sim)sim.alpha(0.2).restart();return;
  }
  if(algo==='random'){VN.forEach(n=>{n.x=Math.random()*W;n.y=Math.random()*H;n.vx=n.vy=0})}
  if(sim){
    const strength=algo==='fa2'?-280:-130;
    const dist=algo==='fa2'?90:65;
    sim.force('charge',d3.forceManyBody().strength(strength))
       .force('link',d3.forceLink(SL).id(d=>d.id).distance(dist).strength(algo==='fa2'?0.6:0.4))
       .alpha(1).restart();
  }
}
function applyFilters(){
  VN=AN.filter(n=>F.types.has(n.node_type)&&n.energy>=F.elo&&n.energy<=F.ehi&&n.depth>=F.dlo&&n.depth<=F.dhi);
  buildSL();compDeg();
  if(sim){sim.nodes(VN);sim.force('link',d3.forceLink(SL).id(d=>d.id).distance(65).strength(0.4));sim.alpha(0.5).restart()}
  updCounts();updBB();
}

// ── Fit view ──────────────────────────────────────────────────────────
function fitView(){
  const ns=VN.filter(n=>n.x!=null);if(!ns.length)return;
  const xs=ns.map(n=>n.x),ys=ns.map(n=>n.y);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const W=cv.width,H=cv.height,pad=60;
  xf.k=Math.min((W-pad*2)/(x1-x0||1),(H-pad*2)/(y1-y0||1),8);
  xf.x=W/2-(x0+x1)/2*xf.k;xf.y=H/2-(y0+y1)/2*xf.k;draw();
}

// ── UI Helpers ────────────────────────────────────────────────────────
function updCounts(){
  document.getElementById('nc').textContent=VN.length;
  document.getElementById('ec').textContent=SL.length;
}
function buildFilterTypes(){
  const cnt={};AN.forEach(n=>{cnt[n.node_type]=(cnt[n.node_type]||0)+1});
  document.getElementById('ftypes').innerHTML=Object.entries(TC).map(([t,c])=>`
    <div class="fi">
      <input type="checkbox" style="accent-color:${c};width:13px;height:13px" ${F.types.has(t)?'checked':''} onchange="toggleType('${t}',this.checked)">
      <div class="fd" style="background:${c}"></div>
      <span class="fl">${t}</span>
      <span class="fc">${cnt[t]||0}</span>
    </div>`).join('');
}
function toggleType(t,on){if(on)F.types.add(t);else if(F.types.size>1)F.types.delete(t);applyFilters()}
function updBB(){
  const cb=document.getElementById('colorBy').value;
  document.getElementById('bbc').textContent=cb;
  document.getElementById('bbs2').textContent=document.getElementById('sizeBy').value==='degree'?'Degree (dynamic)':document.getElementById('sizeBy').value;
  document.getElementById('bbe').textContent=document.getElementById('sizeEdge').value;
  const cl=document.getElementById('bbcl');cl.innerHTML='';
  if(cb==='node_type'){
    const cnt={};AN.forEach(n=>{cnt[n.node_type]=(cnt[n.node_type]||0)+1});
    Object.entries(TC).forEach(([t,c])=>{
      if(!(cnt[t])) return;
      cl.innerHTML+=`<div class="bbli"><div class="bbd" style="background:${c}"></div>${t}<span style="margin-left:auto">${cnt[t]}</span></div>`;
    });
  } else if(cb==='modularity'){
    const cls=[...new Set(VN.map(n=>n.mc||0))].sort((a,b)=>a-b);
    cls.forEach(c=>{cl.innerHTML+=`<div class="bbli"><div class="bbd" style="background:${MC[c%MC.length]}"></div>${c}</div>`});
  }
  const ws=AE.map(e=>e.weight||1);
  document.getElementById('bber').textContent=ws.length?`${Math.min(...ws).toFixed(1)} — ${Math.max(...ws).toFixed(1)}`:'—';
}

// ── Mouse events ──────────────────────────────────────────────────────
cv.addEventListener('mousedown',e=>{
  const h=nAt(e.offsetX,e.offsetY);
  if(h){dragging=true;dn=h;h.fx=h.x;h.fy=h.y}
  else{panning=true;ps={x:e.offsetX-xf.x,y:e.offsetY-xf.y}}
});
cv.addEventListener('mousemove',e=>{
  if(dragging&&dn){const g=s2g(e.offsetX,e.offsetY);dn.fx=g.x;dn.fy=g.y;if(sim)sim.alpha(0.2).restart();draw()}
  else if(panning){xf.x=e.offsetX-ps.x;xf.y=e.offsetY-ps.y;draw()}
  const h=nAt(e.offsetX,e.offsetY),tt=document.getElementById('tt');
  if(h){
    const c=nColor(h),d=deg.get(h.id)||0;
    tt.className='show';tt.style.left=(e.offsetX+14)+'px';tt.style.top=(e.offsetY-14)+'px';
    tt.innerHTML=`<div class="tth" style="color:${c}">${h.node_type}</div>
      <div class="ttr"><span>id</span><span class="ttv" style="font-family:monospace">${h.id.slice(0,12)}…</span></div>
      <div class="ttr"><span>label</span><span class="ttv">${nLabel(h)}</span></div>
      <div class="ttr"><span>energy</span><span class="ttv">${(h.energy||0).toFixed(4)}</span></div>
      <div class="ttr"><span>depth</span><span class="ttv">${(h.depth||0).toFixed(4)}</span></div>
      <div class="ttr"><span>hausdorff</span><span class="ttv">${(h.hausdorff||0).toFixed(4)}</span></div>
      <div class="ttr"><span>degree</span><span class="ttv">${d}</span></div>`;
  }else tt.className='';
});
cv.addEventListener('mouseup',()=>{if(dn){dn.fx=dn.fy=null}dragging=panning=false;dn=null});
cv.addEventListener('dblclick',e=>{
  const h=nAt(e.offsetX,e.offsetY);
  if(h){hl=hl.has(h.id)?new Set():new Set([h.id]);draw()}
});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const f=e.deltaY<0?1.12:1/1.12;
  xf.k=Math.max(0.05,Math.min(20,xf.k*f));
  xf.x=e.offsetX-(e.offsetX-xf.x)*f;xf.y=e.offsetY-(e.offsetY-xf.y)*f;draw();
},{passive:false});

// ── Toolbar ───────────────────────────────────────────────────────────
document.getElementById('rtFit').addEventListener('click',fitView);
document.getElementById('rtZin').addEventListener('click',()=>{
  const W=cv.width,H=cv.height,f=1.3;xf.k=Math.min(xf.k*f,20);xf.x=W/2-(W/2-xf.x)*f;xf.y=H/2-(H/2-xf.y)*f;draw();
});
document.getElementById('rtZout').addEventListener('click',()=>{
  const W=cv.width,H=cv.height,f=1/1.3;xf.k=Math.max(xf.k*f,0.05);xf.x=W/2-(W/2-xf.x)*f;xf.y=H/2-(H/2-xf.y)*f;draw();
});
document.getElementById('rtZrect').addEventListener('click',fitView);

// ── Layout buttons ────────────────────────────────────────────────────
document.querySelectorAll('#algos .la-b').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('#algos .la-b').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');algo=b.dataset.a;
  });
});
document.getElementById('btnRun').addEventListener('click',()=>{if(sim||VN.length)applyLayout()});
document.getElementById('btnStop').addEventListener('click',()=>{if(sim)sim.alpha(0).stop()});

// ── Appearance controls ───────────────────────────────────────────────
document.getElementById('colorBy').addEventListener('change',e=>{AP.colorBy=e.target.value;updBB();draw()});
document.getElementById('sizeBy').addEventListener('change',e=>{
  AP.sizeBy=e.target.value;
  if(sim)sim.force('col',d3.forceCollide().radius(n=>nRad(n)+5)).alpha(0.3).restart();
  updBB();draw();
});
document.getElementById('sizeEdge').addEventListener('change',e=>{AP.sizeEdge=e.target.value;updBB();draw()});
document.getElementById('chkLabels').addEventListener('change',e=>{AP.labels=e.target.checked;draw()});
document.getElementById('chkGlow').addEventListener('change',e=>{AP.glow=e.target.checked;draw()});
document.getElementById('chkEdges').addEventListener('change',e=>{AP.edges=e.target.checked;draw()});

// ── Filter ranges ─────────────────────────────────────────────────────
function setupRange(loId,hiId,loLbl,hiLbl,onChg){
  const lo=document.getElementById(loId),hi=document.getElementById(hiId);
  const ll=document.getElementById(loLbl),hl2=document.getElementById(hiLbl);
  function u(){let l=+lo.value,h=+hi.value;if(l>h)[l,h]=[h,l];ll.textContent=(l/100).toFixed(2);hl2.textContent=(h/100).toFixed(2);onChg(l/100,h/100)}
  lo.addEventListener('input',u);hi.addEventListener('input',u);u();
}
setupRange('felo','fehi','felol','fehil',(l,h)=>{F.elo=l;F.ehi=h;applyFilters()});
setupRange('fdlo','fdhi','fdlol','fdhil',(l,h)=>{F.dlo=l;F.dhi=h;applyFilters()});

// ── Metrics ───────────────────────────────────────────────────────────
function runM(type){
  const el=document.getElementById('mr');el.style.display='block';
  if(type==='degree'){
    compDeg();
    const s=[...deg.entries()].sort((a,b)=>b[1]-a[1]).slice(0,6);
    el.innerHTML='<b>Top degree:</b><br>'+s.map(([id,d])=>`${id.slice(0,10)}: <b>${d}</b>`).join('<br>');
    draw();
  }else if(type==='weighted'){
    const s=[...wdeg.entries()].sort((a,b)=>b[1]-a[1]).slice(0,6);
    el.innerHTML='<b>Top weighted degree:</b><br>'+s.map(([id,d])=>`${id.slice(0,10)}: <b>${d.toFixed(2)}</b>`).join('<br>');
  }else if(type==='modularity'){
    const types=[...new Set(VN.map(n=>n.node_type))];
    VN.forEach(n=>{n.mc=types.indexOf(n.node_type)});
    el.innerHTML=`<b>${types.length} classes</b> (by node_type)<br>`+types.map((t,i)=>`<span style="color:${MC[i%MC.length]}">${t}</span>`).join(', ');
    document.getElementById('colorBy').value='modularity';AP.colorBy='modularity';
    updBB();draw();
  }else if(type==='density'){
    const n=VN.length,e=SL.length;
    const maxE=n*(n-1);const d=maxE>0?(e/maxE*100).toFixed(2):0;
    el.innerHTML=`<b>Nodes:</b> ${n}<br><b>Edges:</b> ${e}<br><b>Density:</b> ${d}%<br><b>Avg degree:</b> ${n>0?(e*2/n).toFixed(2):0}`;
  }
}

// ── Search ────────────────────────────────────────────────────────────
function doSearch(){
  const q=document.getElementById('si').value.trim().toLowerCase();
  const sr=document.getElementById('sr');
  if(!q){sr.style.display='none';hl.clear();draw();return}
  const hits=VN.filter(n=>{
    const s=(nLabel(n)+n.id+n.node_type+JSON.stringify(n.content||'')).toLowerCase();
    return s.includes(q);
  }).slice(0,12);
  hl=new Set(hits.map(n=>n.id));
  sr.style.display=hits.length?'block':'none';
  sr.innerHTML=hits.map(n=>`<div class="sri" onclick="focusN('${n.id}')">
    <div style="width:7px;height:7px;border-radius:50%;background:${nColor(n)};flex-shrink:0"></div>
    <span style="color:var(--dim)">${n.node_type}</span>
    <span>${nLabel(n)}</span>
  </div>`).join('');
  draw();
}
document.getElementById('si').addEventListener('input',doSearch);
document.getElementById('si').addEventListener('keydown',e=>{
  if(e.key==='Escape'){e.target.value='';hl.clear();document.getElementById('sr').style.display='none';draw()}
});
function focusN(id){
  const n=VN.find(n=>n.id===id);if(!n||n.x==null)return;
  const W=cv.width,H=cv.height;xf.k=3;xf.x=W/2-n.x*3;xf.y=H/2-n.y*3;draw();
}

// ── NQL ───────────────────────────────────────────────────────────────
const openNql=()=>document.getElementById('nqlp').classList.add('open');
const closeNql=()=>document.getElementById('nqlp').classList.remove('open');
document.getElementById('btnNql').addEventListener('click',openNql);
document.getElementById('nqlfab').addEventListener('click',openNql);
document.getElementById('nqlClose').addEventListener('click',closeNql);
document.getElementById('nqlRun').addEventListener('click',async()=>{
  const q=document.getElementById('nqli').value.trim();if(!q)return;
  const r=document.getElementById('nqlr');r.textContent='Running…';
  try{
    const res=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nql:q})});
    const d=await res.json();
    if(d.error){r.innerHTML=`<span style="color:#e74c3c">${d.error}</span>`}
    else{
      const ns=d.nodes||[];hl=new Set(ns.map(n=>n.id));
      r.innerHTML=`<span style="color:#4ade80">${ns.length} node(s)</span><br>`+
        ns.slice(0,10).map(n=>`<div style="border-top:1px solid #252535;padding:3px 0">
          <span style="color:${TC[n.node_type]||'#888'}">${n.node_type}</span>
          <span style="color:#666;margin-left:6px;font-family:monospace">${n.id.slice(0,12)}</span>
          <span style="color:#aaa;margin-left:6px">e=${(n.energy||0).toFixed(2)}</span>
        </div>`).join('');
      draw();
    }
  }catch(err){r.innerHTML=`<span style="color:#e74c3c">${err.message}</span>`}
});
document.getElementById('nqlClr').addEventListener('click',()=>{document.getElementById('nqlr').innerHTML='';hl.clear();draw()});

// ── View toggle ───────────────────────────────────────────────────────
let dataNodes=true;
function setView(v){
  const views=[{id:'vg',el:'cw',dsp:'block'},{id:'vd',el:'dv',dsp:'flex'},{id:'va',el:'av',dsp:'flex'}];
  views.forEach(({id,el,dsp})=>{
    document.getElementById(id).classList.toggle('active',id===v);
    document.getElementById(el).style.display=id===v?dsp:'none';
  });
  if(v==='vg'){rsz();draw();}
  if(v==='vd'){renderTable();}
  if(v==='va'){loadAgency();}
}
document.getElementById('vg').addEventListener('click',()=>setView('vg'));
document.getElementById('vd').addEventListener('click',()=>setView('vd'));
document.getElementById('va').addEventListener('click',()=>setView('va'));
document.getElementById('dtNodes').addEventListener('click',()=>{
  dataNodes=true;document.getElementById('dtNodes').classList.add('active');document.getElementById('dtEdges').classList.remove('active');renderTable();
});
document.getElementById('dtEdges').addEventListener('click',()=>{
  dataNodes=false;document.getElementById('dtEdges').classList.add('active');document.getElementById('dtNodes').classList.remove('active');renderTable();
});
function renderTable(){
  const th=document.getElementById('dth'),tb=document.getElementById('dtbd');
  if(dataNodes){
    th.innerHTML='<tr><th>id</th><th>type</th><th>energy</th><th>depth</th><th>hausdorff</th><th>degree</th><th>label / content</th></tr>';
    tb.innerHTML=VN.map(n=>`<tr>
      <td style="font-family:monospace;font-size:11px">${n.id.slice(0,12)}…</td>
      <td><span style="color:${TC[n.node_type]||'#888'}">${n.node_type}</span></td>
      <td>${(n.energy||0).toFixed(4)}</td><td>${(n.depth||0).toFixed(4)}</td>
      <td>${(n.hausdorff||0).toFixed(4)}</td><td>${deg.get(n.id)||0}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px">${nLabel(n)} ${JSON.stringify(n.content||{}).slice(0,80)}</td>
    </tr>`).join('');
  }else{
    th.innerHTML='<tr><th>id</th><th>from</th><th>to</th><th>type</th><th>weight</th></tr>';
    tb.innerHTML=AE.map(e=>`<tr>
      <td style="font-family:monospace;font-size:11px">${e.id.slice(0,12)}…</td>
      <td style="font-family:monospace;font-size:11px">${e.from.slice(0,12)}…</td>
      <td style="font-family:monospace;font-size:11px">${e.to.slice(0,12)}…</td>
      <td>${e.edge_type||'Association'}</td><td>${(e.weight||1).toFixed(3)}</td>
    </tr>`).join('');
  }
}

// ── Load data ─────────────────────────────────────────────────────────
async function loadGraph(){
  document.getElementById('ld').style.display='flex';
  try{
    const r=await fetch('/api/graph');const d=await r.json();
    AN=(d.nodes||[]).map(n=>({...n,x:undefined,y:undefined,vx:0,vy:0,fx:null,fy:null}));
    AE=d.edges||[];
    F.types=new Set(Object.keys(TC));
    VN=[...AN];
    buildSL();compDeg();
    buildFilterTypes();updCounts();updBB();
    rsz();initSim();
    document.getElementById('ld').style.display='none';
    setTimeout(fitView,2600);
  }catch(err){
    document.getElementById('ld').innerHTML=`<div style="color:#e74c3c">Load failed: ${err.message}</div>`;
  }
}
loadGraph();draw();

// ── Agency Panel ──────────────────────────────────────────────────────
async function loadAgency(){
  try{
    // Health
    const hr=await fetch('/api/agency/health/latest').then(r=>r.ok?r.json():null).catch(()=>null);
    if(hr){
      const s=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
      s('ag-energy',hr.mean_energy?.toFixed(3)??'—');
      s('ag-hausdorff',hr.global_hausdorff?.toFixed(3)??'—');
      s('ag-coherence',hr.coherence_score?.toFixed(3)??'—');
      s('ag-gaps',hr.gap_count??'—');
      s('ag-entropy',hr.entropy_spike_count??'—');
      s('ag-fractal',hr.is_fractal?'Yes':'No');
      s('ag-nodes',hr.total_nodes??'—');
      s('ag-edges',hr.total_edges??'—');
      // Color coding
      const ene=document.getElementById('ag-energy');
      if(ene){ene.className='ag-val '+(hr.mean_energy>0.5?'good':hr.mean_energy>0.2?'warn':'bad');}
      const hd=document.getElementById('ag-hausdorff');
      if(hd){hd.className='ag-val '+(hr.is_fractal?'good':'warn');}
    }
    // Observer
    const obs=await fetch('/api/agency/observer').then(r=>r.ok?r.json():null).catch(()=>null);
    if(obs){
      document.getElementById('ag-obs-energy').textContent=obs.energy?.toFixed(3)??'—';
      document.getElementById('ag-obs-depth').textContent=obs.depth?.toFixed(3)??'—';
      document.getElementById('ag-obs-hausdorff').textContent=obs.hausdorff_local?.toFixed(3)??'—';
    }
    // Desires
    const dr=await fetch('/api/agency/desires').then(r=>r.ok?r.json():null).catch(()=>null);
    const dl=document.getElementById('ag-desire-list');
    if(dr&&dr.desires){
      dl.innerHTML=dr.desires.length===0?'<div style="color:var(--dim);font-size:12px;padding:8px">No pending desires</div>':
        dr.desires.slice(0,10).map(d=>`<div class="ag-item">
          <div><span style="color:var(--dim);font-size:10px">Sector (${d.sector.depth_bin},${d.sector.angular_bin})</span><br>
          <span style="font-size:11px">${d.suggested_query.slice(0,60)}</span></div>
          <span class="ag-pri">${d.priority.toFixed(2)}</span>
        </div>`).join('');
    }else{dl.innerHTML='<div style="color:var(--dim);font-size:12px;padding:8px">No data</div>';}
    // Evolution
    const ev=await fetch('/api/agency/evolution').then(r=>r.ok?r.json():null).catch(()=>null);
    if(ev){
      document.getElementById('ag-evo-gen').textContent=ev.generation??0;
      document.getElementById('ag-evo-strat').textContent=ev.last_strategy??'—';
      const eh=document.getElementById('ag-evo-history');
      if(ev.fitness_history&&ev.fitness_history.length>0){
        eh.innerHTML=ev.fitness_history.slice(-5).reverse().map(f=>`<div class="ag-item">
          <span>Gen ${f.generation}: ${f.strategy}</span>
          <span class="ag-pri">${f.fitness.toFixed(3)}</span>
        </div>`).join('');
      }else{eh.innerHTML='<div style="color:var(--dim);font-size:12px;padding:4px">No history yet</div>';}
    }
    // Narrative
    const nr=await fetch('/api/agency/narrative').then(r=>r.ok?r.json():null).catch(()=>null);
    const nt=document.getElementById('ag-narr-text');
    if(nr&&nr.narrative){nt.textContent=nr.narrative;}
    else{nt.textContent='No narrative generated yet. Agency engine needs to tick with a health report.';}
  }catch(e){console.error('agency load error',e);}
}
</script>
</body>
</html>"##;
