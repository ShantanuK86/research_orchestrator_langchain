/**
 * ui.js — all DOM mutations live here
 */

export const AGENT_STYLES = {
  supervisor: { bg:'rgba(0,212,170,0.1)',  color:'#00d4aa', label:'SUPERVISOR' },
  search:     { bg:'rgba(77,158,255,0.1)', color:'#4d9eff', label:'SEARCH'     },
  critic:     { bg:'rgba(255,107,107,0.1)',color:'#ff6b6b', label:'CRITIC'     },
  writer:     { bg:'rgba(245,166,35,0.1)', color:'#f5a623', label:'WRITER'     },
  system:     { bg:'rgba(255,255,255,0.05)',color:'#6b7a8d',label:'SYS'        },
};

let _logCount  = 0;
let _startTime = null;
let _timerInt  = null;
let _scoreHistory = [];

const el = id => document.getElementById(id);

function elapsed() {
  if (!_startTime) return '0s';
  return Math.floor((Date.now() - _startTime) / 1000) + 's';
}

// ── Exports ──────────────────────────────────────────────

export function showMainPanel() { el('mainPanel').classList.add('visible'); }

export function resetUI() {
  _logCount = 0; _scoreHistory = [];
  el('logBody').innerHTML     = '';
  el('logCount').textContent  = '0 events';
  el('metricIterations').textContent = '0';
  el('metricMessages').textContent   = '0';
  el('metricTokens').textContent     = '0';
  el('metricTime').textContent       = '0s';
  el('iterLabel').textContent        = '0 / 3 max';
  el('resultSection').style.display  = 'none';
  el('resultText').innerHTML         = '';
  el('scoreRings').innerHTML         = '';
  el('loopIndicator').classList.remove('visible');

  for (let i = 0; i < 5; i++) {
    const d = el(`idot-${i}`);
    if (d) d.className = 'iter-dot';
  }
  ['supervisor','search','critic','writer'].forEach(a => {
    setNodeState(a,'idle'); setLegendActive(a,false);
    const it = el('iter-'+a); if(it) it.textContent='';
  });
  // reset arrows
  document.querySelectorAll('.arrow-line').forEach(a => a.classList.remove('active'));
}

export function startTimer() {
  _startTime = Date.now();
  _timerInt = setInterval(() => { el('metricTime').textContent = elapsed(); }, 500);
}

export function stopTimer() {
  clearInterval(_timerInt);
  el('metricTime').textContent = elapsed();
}

export function addLog(agent, message, thinking=false, type='log') {
  _logCount++;
  el('logCount').textContent = _logCount + ' events';
  el('metricMessages').textContent = _logCount;

  const body = el('logBody');
  const s = AGENT_STYLES[agent] || AGENT_STYLES.system;

  if (type === 'thought') {
    const bubble = document.createElement('div');
    bubble.className = 'thought-bubble';
    bubble.textContent = '💭 ' + message;
    body.appendChild(bubble);
  } else {
    const entry = document.createElement('div');
    entry.className = `log-entry${type==='thought'?' thought-entry':''}`;
    entry.innerHTML = `
      <span class="log-agent-tag" style="background:${s.bg};color:${s.color}">${s.label}</span>
      <div class="log-message${thinking?' thinking':''}">${message}</div>
      <span class="log-time">${elapsed()}</span>
    `;
    body.appendChild(entry);
  }
  body.scrollTop = body.scrollHeight;
}

export function setNodeState(node, state, iterText='') {
  const nodeEl   = el('node-'+node);
  const statusEl = el('status-'+node);
  const iterEl   = el('iter-'+node);
  if (!nodeEl) return;
  nodeEl.className = 'gnode';
  if (state==='active') nodeEl.classList.add('active');
  else if (state==='done')  nodeEl.classList.add('done');
  else if (state==='error') nodeEl.classList.add('error');
  if (statusEl) statusEl.textContent = state;
  if (iterEl && iterText) iterEl.textContent = iterText;
}

export function setLegendActive(agent, active) {
  const el2 = el('legend-'+agent);
  if (!el2) return;
  if (active) el2.classList.add('active-agent');
  else el2.classList.remove('active-agent');
}

export function activateArrow(from, to) {
  // map node pair → arrow index
  const map = { 'supervisor-search':0, 'search-critic':1, 'critic-writer':2 };
  const key = `${from}-${to}`;
  document.querySelectorAll('.arrow-line').forEach(a => a.classList.remove('active'));
  const idx = map[key];
  if (idx !== undefined) {
    const arrows = document.querySelectorAll('.arrow-line');
    if (arrows[idx]) arrows[idx].classList.add('active');
  }
}

export function showLoopIndicator(visible, iteration=0) {
  const el2 = el('loopIndicator');
  if (!el2) return;
  if (visible) {
    el2.classList.add('visible');
    el2.querySelector('.loop-text').textContent =
      `Critic rejected → loop ${iteration}`;
  } else {
    el2.classList.remove('visible');
  }
}

export function updateIteration(n, rejected=false) {
  el('metricIterations').textContent = n;
  el('iterLabel').textContent = `${n} / 3 max`;
  for (let i=0;i<5;i++) {
    const dot = el(`idot-${i}`); if(!dot) continue;
    dot.className='iter-dot';
    if (i<n-1)     dot.classList.add('done');
    else if (i===n-1) dot.classList.add(rejected?'rejected':'active');
  }
}

export function updateTokens(n) {
  el('metricTokens').textContent = n;
  el('metricTokens').closest('.stat-item')?.classList.add('highlight');
  setTimeout(()=> el('metricTokens').closest('.stat-item')?.classList.remove('highlight'), 600);
}

export function addScoreRing(score, iteration) {
  _scoreHistory.push({score, iteration});
  const container = el('scoreRings');
  const circumference = 2 * Math.PI * 16; // r=16
  const offset = circumference - (score/10)*circumference;
  const color = score>=7 ? '#00d4aa' : score>=5 ? '#f5a623' : '#ff6b6b';

  const wrap = document.createElement('div');
  wrap.className = 'score-ring-wrap';
  wrap.title = `Iteration ${iteration}: ${score}/10`;
  wrap.innerHTML = `
    <svg viewBox="0 0 38 38" width="44" height="44">
      <circle class="score-ring-bg" cx="19" cy="19" r="16"/>
      <circle class="score-ring-fill" cx="19" cy="19" r="16"
        stroke="${color}"
        stroke-dasharray="${circumference}"
        stroke-dashoffset="${circumference}"
        style="transition:stroke-dashoffset .8s ease"/>
    </svg>
    <div class="score-ring-num" style="color:${color}">${score}</div>
  `;
  container.appendChild(wrap);
  // Animate fill
  requestAnimationFrame(()=>{
    requestAnimationFrame(()=>{
      wrap.querySelector('.score-ring-fill').style.strokeDashoffset = offset;
    });
  });
}

export function setRunButton(state) {
  const btn = el('runBtn');
  if (state==='loading') {
    btn.disabled=true;
    btn.innerHTML='<div class="spinner"></div> Running...';
  } else {
    btn.disabled=false;
    btn.innerHTML='<span>▶</span> Run Again';
  }
}

export function renderReport(markdown) {
  el('resultSection').style.display='block';
  let html = markdown
    .replace(/^## (.+)$/gm,   '<h2>$1</h2>')
    .replace(/^### (.+)$/gm,  '<h3>$1</h3>')
    .replace(/^# (.+)$/gm,    '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,    '<em>$1</em>')
    .replace(/`(.+?)`/g,      '<code>$1</code>')
    .replace(/^> (.+)$/gm,    '<blockquote>$1</blockquote>')
    .replace(/^- (.+)$/gm,    '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm,'<li>$2</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>')
    .replace(/\n\n/g,'</p><p>');
  html = html.replace(/<p><\/p>/g,'').replace(/<p>(<h[23])/g,'$1').replace(/(<\/h[23]>)<\/p>/g,'$1');
  el('resultText').innerHTML = html;
  el('resultSection').scrollIntoView({behavior:'smooth',block:'start'});
}
