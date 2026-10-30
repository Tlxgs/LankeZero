import { BOARD_RULES } from './go.mjs';

const $ = (sel) => document.querySelector(sel);
const COLORS = { black: 1, white: -1 };
const SIZE_RANGE = [9, 13, 19];

const state = {
  boardSize: 19,
  humanColor: 1,
  snap: null,
  modelReady: false,
  modelProgress: 0,
  backend: '',
  search: null,
  error: '',
  hover: -1,
  toastTimer: null,
};

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem('lanke.settings') || '{}');
    if (SIZE_RANGE.includes(Number(s.boardSize))) state.boardSize = Number(s.boardSize);
    if (s.humanColor === 1 || s.humanColor === -1) state.humanColor = s.humanColor;
  } catch (_) { /* ignore */ }
}
function saveSettings() {
  try {
    localStorage.setItem('lanke.settings', JSON.stringify({
      boardSize: state.boardSize,
      humanColor: state.humanColor,
    }));
  } catch (_) { /* ignore */ }
}

const canvas = $('#board');
const ctx = canvas.getContext('2d');
let layout = { size: 0, margin: 0, cell: 0 };

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const size = Math.max(200, Math.min(rect.width, rect.height || rect.width));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const n = state.snap ? state.snap.boardSize : state.boardSize;
  const margin = size * (n >= 19 ? 0.065 : 0.075);
  layout = { size, margin, cell: (size - 2 * margin) / (n - 1) };
  draw();
}

function stoneRadius() {
  return layout.cell * 0.435;
}

function drawStone(x, y, color, alpha = 1) {
  const r = stoneRadius();
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (color === 1) {
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.38, r * 0.1, x, y, r);
    g.addColorStop(0, '#4b4a45');
    g.addColorStop(1, '#1d1c1a');
    ctx.fillStyle = g;
    ctx.shadowColor = 'rgba(30,28,24,.28)';
    ctx.shadowBlur = Math.max(2, r * 0.22);
    ctx.shadowOffsetY = Math.max(1, r * 0.08);
  } else {
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
    g.addColorStop(0, '#fbfaf6');
    g.addColorStop(1, '#d9d5cb');
    ctx.fillStyle = g;
    ctx.shadowColor = 'rgba(30,28,24,.22)';
    ctx.shadowBlur = Math.max(2, r * 0.2);
    ctx.shadowOffsetY = Math.max(1, r * 0.08);
  }
  ctx.fill();
  ctx.restore();
}

function draw() {
  const snap = state.snap;
  const n = snap ? snap.boardSize : state.boardSize;
  const { size, margin, cell } = layout;
  if (!size) return;
  ctx.clearRect(0, 0, size, size);
  ctx.save();

  // 棋盘线
  ctx.strokeStyle = '#706d64';
  ctx.lineWidth = Math.max(0.8, cell * 0.018);
  ctx.lineCap = 'square';
  const x0 = margin;
  const x1 = margin + cell * (n - 1);
  for (let i = 0; i < n; i++) {
    const p = margin + i * cell;
    ctx.beginPath();
    ctx.moveTo(x0, p);
    ctx.lineTo(x1, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, x0);
    ctx.lineTo(p, x1);
    ctx.stroke();
  }

  // 星位：13/19 路 3x3，9 路五子。
  const stars = n === 9
    ? [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]]
    : (n === 13
      ? [[3, 3], [3, 6], [3, 9], [6, 3], [6, 6], [6, 9], [9, 3], [9, 6], [9, 9]]
      : [[3, 3], [3, 9], [3, 15], [9, 3], [9, 9], [9, 15], [15, 3], [15, 9], [15, 15]]);
  for (const [r, c] of stars) {
    const x = margin + c * cell;
    const y = margin + r * cell;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1.4, cell * 0.065), 0, Math.PI * 2);
    ctx.fillStyle = '#5f5d56';
    ctx.fill();
  }

  // 棋子
  if (snap && snap.board && snap.board.length) {
    for (let idx = 0; idx < snap.board.length; idx++) {
      const v = snap.board[idx];
      if (!v) continue;
      const r = (idx / n) | 0;
      const c = idx % n;
      drawStone(margin + c * cell, margin + r * cell, v);
    }
    // 上一手
    if (snap.lastMove >= 0 && snap.lastMove < n * n) {
      const r = (snap.lastMove / n) | 0;
      const c = snap.lastMove % n;
      const v = snap.board[snap.lastMove];
      ctx.beginPath();
      ctx.arc(margin + c * cell, margin + r * cell, Math.max(1.8, cell * 0.085), 0, Math.PI * 2);
      ctx.fillStyle = v === 1 ? '#e8e5dc' : '#302e2a';
      ctx.fill();
    }
  }

  // 悬停预览
  if (state.hover >= 0 && canHumanPlay() && snap && snap.board[state.hover] === 0) {
    const r = (state.hover / n) | 0;
    const c = state.hover % n;
    drawStone(margin + c * cell, margin + r * cell, state.humanColor, 0.28);
  }
  ctx.restore();
}

function canHumanPlay() {
  const s = state.snap;
  return !!s && !s.gameOver && !s.thinking && s.currentPlayer === s.humanColor && state.modelReady;
}

function pointFromEvent(ev) {
  const s = state.snap;
  if (!s) return -1;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  const { margin, cell } = layout;
  const n = s.boardSize;
  const c = Math.round((x - margin) / cell);
  const r = Math.round((y - margin) / cell);
  if (r < 0 || r >= n || c < 0 || c >= n) return -1;
  const cx = margin + c * cell;
  const cy = margin + r * cell;
  if ((x - cx) ** 2 + (y - cy) ** 2 > (cell * 0.62) ** 2) return -1;
  return r * n + c;
}

function statusText() {
  const s = state.snap;
  if (state.error) return state.error;
  if (!state.modelReady) {
    return state.modelProgress > 0 ? ('加载模型 ' + state.modelProgress + '%') : '准备模型';
  }
  if (!s) return '准备中';
  if (s.gameOver) {
    if (s.winner === 0) return '和棋';
    const who = s.winner === 1 ? '黑' : '白';
    return who + '胜 · ' + Math.abs(s.finalPoints).toFixed(1) + ' 目';
  }
  if (s.thinking) {
    if (state.search) return 'AI 思考中 · ' + state.search.done + '/' + state.search.total;
    return 'AI 思考中';
  }
  if (s.currentPlayer === s.humanColor) return '轮到你 · ' + (s.humanColor === 1 ? '黑' : '白');
  return 'AI 准备中';
}

function renderStatus() {
  const el = $('#status');
  el.textContent = statusText();
  el.classList.toggle('is-thinking', !!(state.snap && state.snap.thinking));
  el.classList.toggle('is-error', !!state.error);
  const pass = $('#pass');
  const s = state.snap;
  pass.disabled = !canHumanPlay() || !s || !s.legal.includes(s.boardSize * s.boardSize);
  $('#new').disabled = false;
  const colorSeg = $('#color');
  for (const btn of colorSeg.querySelectorAll('button')) {
    const active = Number(btn.dataset.color) === state.humanColor;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
  const sizeSeg = $('#size');
  for (const btn of sizeSeg.querySelectorAll('button')) {
    const active = Number(btn.dataset.size) === state.boardSize;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
  canvas.style.cursor = canHumanPlay() ? 'pointer' : 'default';
  document.body.classList.toggle('is-thinking', !!(s && s.thinking));
}

function showToast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

function onSnapshot(snap) {
  state.snap = snap;
  state.error = '';
  state.hover = -1;
  state.search = null;
  if (snap.boardSize !== state.boardSize) {
    state.boardSize = snap.boardSize;
    saveSettings();
  }
  fitCanvas();
  renderStatus();
}

const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'model') {
    state.modelProgress = msg.progress || 0;
    if (msg.state === 'ready') {
      state.modelReady = true;
      state.backend = msg.backend || '';
    }
    renderStatus();
  } else if (msg.type === 'snapshot') {
    onSnapshot(msg);
  } else if (msg.type === 'search') {
    state.search = { done: msg.done, total: msg.total };
    renderStatus();
  } else if (msg.type === 'error') {
    state.error = msg.message || '出错了';
    if (state.modelReady) state.error = '出错了：' + state.error;
    renderStatus();
    showToast(state.error);
  }
};

function sendNew() {
  state.search = null;
  state.error = '';
  state.snap = state.snap ? { ...state.snap, thinking: true, legal: [] } : null;
  draw();
  renderStatus();
  worker.postMessage({ type: 'new', boardSize: state.boardSize, humanColor: state.humanColor });
}

function init() {
  loadSettings();
  for (const btn of $('#color').querySelectorAll('button')) {
    btn.addEventListener('click', () => {
      const v = Number(btn.dataset.color);
      if (v === state.humanColor) return;
      state.humanColor = v;
      saveSettings();
      sendNew();
    });
  }
  for (const btn of $('#size').querySelectorAll('button')) {
    btn.addEventListener('click', () => {
      const v = Number(btn.dataset.size);
      if (v === state.boardSize) return;
      state.boardSize = v;
      saveSettings();
      sendNew();
    });
  }
  $('#pass').addEventListener('click', () => {
    if (canHumanPlay()) worker.postMessage({ type: 'pass' });
  });
  $('#new').addEventListener('click', () => sendNew());

  canvas.addEventListener('mousemove', (ev) => {
    const p = pointFromEvent(ev);
    const s = state.snap;
    const next = (p >= 0 && s && s.legal.includes(p)) ? p : -1;
    if (next !== state.hover) {
      state.hover = next;
      draw();
    }
  });
  canvas.addEventListener('mouseleave', () => {
    if (state.hover !== -1) {
      state.hover = -1;
      draw();
    }
  });
  canvas.addEventListener('click', (ev) => {
    if (!canHumanPlay()) return;
    const p = pointFromEvent(ev);
    if (p < 0) return;
    const s = state.snap;
    if (!s.legal.includes(p)) return;
    state.hover = -1;
    worker.postMessage({ type: 'move', move: p });
  });

  const ro = new ResizeObserver(() => fitCanvas());
  ro.observe(canvas);
  fitCanvas();
  renderStatus();

  const params = new URLSearchParams(location.search);
  const modelUrl = params.get('model') || window.LANKE_MODEL_URL || new URL('./web/model/model.onnx', location.href).href;
  worker.postMessage({ type: 'init', boardSize: state.boardSize, humanColor: state.humanColor, modelUrl });
}

init();
