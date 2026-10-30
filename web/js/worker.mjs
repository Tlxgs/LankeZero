// 演示页的推理与搜索 worker。主线程只负责画棋盘和收发消息。
import * as ort from '../vendor/ort/ort.webgpu.bundle.min.mjs';
import { GoGame, BOARD_RULES, INPUT_CHANNELS } from './go.mjs';
import { MCTS } from './mcts.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';
if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance';

const NUM_SIMULATIONS = 100;
const BATCH_SIZE = 8;
const KOMI = 7.5;

let session = null;
let backend = '';
let modelUrl = '';
let boardSize = 19;
let humanColor = 1;
let game = null;
let mcts = null;
let generation = 0;
let activeSearch = null;
let pendingRestart = false;

const post = (msg) => self.postMessage(msg);

function snapshot(extra = {}) {
  const legal = game ? game.getLegalMoves() : [];
  const data = {
    type: 'snapshot',
    board: game ? Array.from(game.board) : [],
    boardSize,
    humanColor,
    currentPlayer: game ? game.currentPlayer : 1,
    lastMove: game ? game.lastMove : -2,
    lastWasPass: game ? game.lastMove === game.passIndex : false,
    moveCount: game ? game.moveCount : 0,
    legal,
    gameOver: game ? game.gameOver : false,
    winner: game ? game.winner : null,
    finalPoints: game ? game.finalPoints : 0,
    thinking: !!activeSearch || pendingRestart,
    backend,
  };
  if (game && game.gameOver) {
    const s = game.score();
    data.score = s;
  }
  post({ ...data, ...extra });
}

function refresh() {
  snapshot();
}

async function loadModel(url) {
  if (session && modelUrl === url) return;
  modelUrl = url;
  session = null;
  backend = '';
  post({ type: 'model', state: 'loading', progress: 0, backend: '' });
  let res;
  try {
    res = await fetch(url, { cache: 'default' });
  } catch (e) {
    throw new Error('无法读取模型文件。请通过 HTTP 打开页面，并确认 web/model/model.onnx 存在。');
  }
  if (!res.ok) throw new Error('模型文件读取失败：HTTP ' + res.status);
  const total = Number(res.headers.get('content-length') || 0);
  let bytes;
  if (res.body && total > 0) {
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      post({ type: 'model', state: 'loading', progress: Math.min(99, Math.round(got / total * 100)), backend: '' });
    }
    bytes = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.length;
    }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
  }
  post({ type: 'model', state: 'loading', progress: 100, backend: '' });
  let lastErr = null;
  if (self.navigator && navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      backend = 'WebGPU';
    } catch (e) {
      lastErr = e;
      session = null;
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    backend = 'WASM';
  }
  post({ type: 'model', state: 'ready', progress: 100, backend, warning: lastErr ? String(lastErr) : '' });
}

function makeMcts() {
  return new MCTS({
    boardSize,
    evaluate,
    numSimulations: NUM_SIMULATIONS,
    batchSize: BATCH_SIZE,
    cPuct: 1.0,
    cPuctLog: 0.45,
    cPuctBase: 500,
    utilStdevPrior: 0.05,
    utilStdevScale: 0.85,
    fpuReduction: 0.05,
    virtualLoss: 5,
    candidateMass: 0.95,
    minCandidates: 8,
    alpha: 0.5,
    temperature: 0,
  });
}

async function evaluate(states, masks) {
  if (!session) throw new Error('模型尚未加载');
  const b = states.length;
  const n = boardSize;
  const nn = n * n;
  const total = nn + 1;
  const planes = states[0].length;
  if (planes !== INPUT_CHANNELS * n * n) throw new Error('输入张量尺寸不一致');
  const data = new Float32Array(b * planes);
  for (let i = 0; i < b; i++) data.set(states[i], i * planes);
  const input = new ort.Tensor('float32', data, [b, INPUT_CHANNELS, n, n]);
  const out = await session.run({ state: input });
  const logits = out.policy.data;
  const values = out.value.data;
  const wins = out.win.data;
  const policies = [];
  for (let i = 0; i < b; i++) {
    const p = new Float32Array(total);
    let max = -Infinity;
    for (let j = 0; j < total; j++) {
      if (masks[i][j]) {
        const v = logits[i * total + j];
        p[j] = v;
        if (v > max) max = v;
      } else {
        p[j] = -1e9;
      }
    }
    let sum = 0;
    for (let j = 0; j < total; j++) {
      if (masks[i][j]) {
        const v = Math.exp(p[j] - max);
        p[j] = v;
        sum += v;
      } else {
        p[j] = 0;
      }
    }
    if (sum > 0) for (let j = 0; j < total; j++) if (masks[i][j]) p[j] /= sum;
    policies.push(p);
  }
  return { policies, values: Array.from(values), wins: Array.from(wins) };
}

async function startAiTurn() {
  if (!game || game.gameOver || game.currentPlayer === humanColor) return;
  if (activeSearch) {
    pendingRestart = true;
    return;
  }
  const myGen = generation;
  const token = { cancelled: false };
  activeSearch = token;
  snapshot();
  try {
    if (!mcts.root) await mcts.initRoot(game);
    if (myGen !== generation) return;
    const result = await mcts.search(
      game,
      NUM_SIMULATIONS,
      (done, total) => post({ type: 'search', done, total }),
      () => myGen !== generation,
    );
    if (!result || myGen !== generation) return;
    let move = result.move;
    if (!game.isLegal(move)) {
      const legal = game.getLegalMoves();
      if (!legal.length) throw new Error('没有合法着法');
      move = legal[0];
    }
    if (!game.play(move)) throw new Error('AI 落子失败');
    if (myGen !== generation) return;
    snapshot();
    await mcts.updateRoot(game, move);
    if (game.gameOver) return;
  } catch (e) {
    if (myGen === generation) post({ type: 'error', message: String(e && e.message || e) });
  } finally {
    if (activeSearch === token) activeSearch = null;
    if (pendingRestart) {
      pendingRestart = false;
      queueMicrotask(() => startAiTurn());
    } else if (myGen === generation && game && !game.gameOver) {
      snapshot();
    }
  }
}

async function newGame(n, color) {
  generation++;
  boardSize = n;
  humanColor = color;
  if (activeSearch) pendingRestart = true;
  game = new GoGame(boardSize, KOMI);
  mcts = makeMcts();
  snapshot();
  if (!activeSearch) startAiTurn();
}

async function onMove(move) {
  if (!game || game.gameOver || activeSearch) return;
  if (game.currentPlayer !== humanColor) return;
  const myGen = generation;
  const g = game;
  const m = mcts;
  if (!g.play(move)) {
    snapshot({ error: '这一手不合法' });
    return;
  }
  snapshot();
  if (!g.gameOver) {
    try {
      await m.updateRoot(g, move);
    } catch (e) {
      // 树推进失败不致命，搜索会重新建根。
      m.reset();
    }
    if (myGen === generation && g.currentPlayer !== humanColor) startAiTurn();
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      await loadModel(msg.modelUrl);
      await newGame(msg.boardSize || 19, msg.humanColor || 1);
    } else if (msg.type === 'new') {
      await newGame(msg.boardSize || 19, msg.humanColor || 1);
    } else if (msg.type === 'move') {
      await onMove(msg.move);
    } else if (msg.type === 'pass') {
      if (!game || activeSearch) return;
      await onMove(game.passIndex);
    }
  } catch (e) {
    post({ type: 'error', message: String(e && e.message || e) });
  }
};
