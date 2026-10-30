// 9/13/19 路围棋规则与网络输入编码。纯 JS，无依赖。
// 与 python/game.py 保持同一套规则：简单劫、自杀禁手、Tromp-Taylor 数目 + 小块清理。

export const KOMI = 7.5;
export const PASS_MOVE = -1;
export const BOARD_RULES = {
  9: { maxMoves: 180, minMoves: 30, passLimit: 40 },
  13: { maxMoves: 300, minMoves: 100, passLimit: 60 },
  19: { maxMoves: 420, minMoves: 320, passLimit: 100 },
};

export const CH_OWN_STONES = 0;
export const CH_OPP_STONES = 1;
export const CH_LAST_MOVE = 2;
export const CH_KOMI = 3;
export const CH_OWN_URGENCY = 4;
export const CH_OPP_URGENCY = 5;
export const CH_LEGAL_MASK = 6;
export const CH_OWN_EYES = 7;
export const CH_OPP_EYES = 8;
export const CH_BOUNDARY = 9;
export const INPUT_CHANNELS = 10;

const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];

function inBoard(r, c, n) {
  return r >= 0 && r < n && c >= 0 && c < n;
}

function boardsEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function collectGroup(board, n, start, visited) {
  const color = board[start];
  const stones = [start];
  const stack = [start];
  const libSeen = new Uint8Array(n * n);
  let libs = 0;
  visited[start] = 1;
  while (stack.length) {
    const cur = stack.pop();
    const r = (cur / n) | 0;
    const c = cur % n;
    for (let k = 0; k < 4; k++) {
      const nr = r + DR[k];
      const nc = c + DC[k];
      if (!inBoard(nr, nc, n)) continue;
      const ni = nr * n + nc;
      const v = board[ni];
      if (v === 0) {
        if (!libSeen[ni]) {
          libSeen[ni] = 1;
          libs++;
        }
      } else if (v === color && !visited[ni]) {
        visited[ni] = 1;
        stones.push(ni);
        stack.push(ni);
      }
    }
  }
  return { stones, libs };
}

// 在 board 上尝试落子。成功返回被提子位置；失败返回 null，并保证 board 复原。
function applyMoveRaw(board, n, idx, player, prevBoard) {
  if (idx < 0 || idx >= n * n || board[idx] !== 0) return null;
  const before = board.slice();
  const r = (idx / n) | 0;
  const c = idx % n;
  board[idx] = player;
  const captured = [];
  const visited = new Uint8Array(n * n);
  for (let k = 0; k < 4; k++) {
    const nr = r + DR[k];
    const nc = c + DC[k];
    if (!inBoard(nr, nc, n)) continue;
    const ni = nr * n + nc;
    if (board[ni] === -player && !visited[ni]) {
      const g = collectGroup(board, n, ni, visited);
      if (g.libs === 0) {
        for (const p of g.stones) {
          board[p] = 0;
          captured.push(p);
        }
      }
    }
  }
  if (captured.length === 0) {
    const own = collectGroup(board, n, idx, new Uint8Array(n * n));
    if (own.libs === 0) {
      board.set(before);
      return null;
    }
  }
  if (captured.length > 0 && prevBoard && boardsEqual(board, prevBoard)) {
    board.set(before);
    return null;
  }
  return { captured };
}

function calcKoPoint(board, n, lastIdx, captured, player) {
  if (captured.length !== 1) return -1;
  const cp = captured[0];
  const copy = board.slice();
  const res = applyMoveRaw(copy, n, cp, -player, null);
  if (res && res.captured.length === 1 && res.captured[0] === lastIdx) return cp;
  return -1;
}

// 单点真眼；与 game.py 的 _eye_color 同判据，边界/角自动收紧。
function eyeColor(board, n, r, c) {
  let col = 0;
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (const [dr, dc] of dirs) {
    const nr = r + dr;
    const nc = c + dc;
    if (!inBoard(nr, nc, n)) continue;
    const v = board[nr * n + nc];
    if (v === 0) return 0;
    if (col === 0) col = v;
    else if (v !== col) return 0;
  }
  if (col === 0) return 0;
  let nd = 0;
  let enemy = 0;
  let nonown = 0;
  for (const [dr, dc] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
    const nr = r + dr;
    const nc = c + dc;
    if (!inBoard(nr, nc, n)) continue;
    nd++;
    const v = board[nr * n + nc];
    if (v === -col) {
      enemy++;
      nonown++;
    } else if (v === 0) {
      nonown++;
    }
  }
  if (nd === 4) return enemy <= 1 ? col : 0;
  return nonown === 0 ? col : 0;
}

function findRoot(parent, x) {
  let r = x;
  while (parent[r] !== r) {
    parent[r] = parent[parent[r]];
    r = parent[r];
  }
  return r;
}

function unionRoot(parent, a, b) {
  a = findRoot(parent, a);
  b = findRoot(parent, b);
  if (a !== b) parent[a] = b;
}

// 把四个特征平面写进 out（横盘优先，与 PyTorch NCHW 展平一致）。
function groupChannels(board, n, player, out, base) {
  const nn = n * n;
  const labels = new Int32Array(nn);
  const libSeen = new Int32Array(nn);
  const lib = new Int32Array(nn + 1);
  const groupColor = new Int8Array(nn + 1);
  const stack = new Int32Array(nn);
  let nblk = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const start = r * n + c;
      const color = board[start];
      if (color === 0 || labels[start] !== 0) continue;
      const gid = ++nblk;
      groupColor[gid] = color;
      let top = 0;
      stack[top++] = start;
      labels[start] = gid;
      while (top > 0) {
        const cur = stack[--top];
        const cr = (cur / n) | 0;
        const cc = cur % n;
        for (let k = 0; k < 4; k++) {
          const nr = cr + DR[k];
          const nc = cc + DC[k];
          if (!inBoard(nr, nc, n)) continue;
          const ni = nr * n + nc;
          const nb = board[ni];
          if (nb === 0) {
            if (libSeen[ni] !== gid) {
              libSeen[ni] = gid;
              lib[gid]++;
            }
          } else if (nb === color && labels[ni] === 0) {
            labels[ni] = gid;
            stack[top++] = ni;
          }
        }
      }
    }
  }

  const regLab = new Int32Array(nn).fill(-1);
  const regStart = new Int32Array(nn + 1);
  const regSize = new Int32Array(nn + 1);
  const regPts = new Int32Array(nn);
  let nreg = 0;
  let npos = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const start = r * n + c;
      if (board[start] !== 0 || regLab[start] >= 0) continue;
      const rid = ++nreg;
      regStart[rid] = npos;
      let top = 0;
      stack[top++] = start;
      regLab[start] = rid;
      let cnt = 0;
      while (top > 0) {
        const cur = stack[--top];
        regPts[npos++] = cur;
        cnt++;
        const cr = (cur / n) | 0;
        const cc = cur % n;
        for (let k = 0; k < 4; k++) {
          const nr = cr + DR[k];
          const nc = cc + DC[k];
          if (!inBoard(nr, nc, n)) continue;
          const ni = nr * n + nc;
          if (board[ni] === 0 && regLab[ni] < 0) {
            regLab[ni] = rid;
            stack[top++] = ni;
          }
        }
      }
      regSize[rid] = cnt;
    }
  }

  const parent = new Int32Array(nblk + 1);
  for (let i = 0; i <= nblk; i++) parent[i] = i;
  // 共享 >=2 个空点的同色块并组（对手一手切不断）。
  const pairCount = new Map();
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const idx = r * n + c;
      if (board[idx] !== 0) continue;
      const seen = [];
      for (let k = 0; k < 4; k++) {
        const nr = r + DR[k];
        const nc = c + DC[k];
        if (!inBoard(nr, nc, n)) continue;
        const gid = labels[nr * n + nc];
        if (gid === 0 || seen.includes(gid)) continue;
        seen.push(gid);
      }
      for (let i = 0; i < seen.length; i++) {
        for (let j = i + 1; j < seen.length; j++) {
          const a = seen[i];
          const b = seen[j];
          // 只并同色块。
          if (groupColor[a] !== groupColor[b]) continue;
          const key = a < b ? a * (nblk + 1) + b : b * (nblk + 1) + a;
          pairCount.set(key, (pairCount.get(key) || 0) + 1);
        }
      }
    }
  }
  for (const [key, cnt] of pairCount) {
    if (cnt < 2) continue;
    const a = (key / (nblk + 1)) | 0;
    const b = key % (nblk + 1);
    unionRoot(parent, a, b);
  }

  const grpEye = new Int32Array(nblk + 1);
  for (let rid = 1; rid <= nreg; rid++) {
    const s0 = regStart[rid];
    const e0 = s0 + regSize[rid];
    let col = 0;
    let bad = false;
    for (let i = s0; i < e0; i++) {
      const cur = regPts[i];
      const cr = (cur / n) | 0;
      const cc = cur % n;
      for (let k = 0; k < 4; k++) {
        const nr = cr + DR[k];
        const nc = cc + DC[k];
        if (!inBoard(nr, nc, n)) continue;
        const v = board[nr * n + nc];
        if (v === 0) continue;
        if (col === 0) col = v;
        else if (v !== col) {
          bad = true;
          break;
        }
      }
      if (bad) break;
    }
    if (bad || col === 0) continue;
    let e;
    if (regSize[rid] === 1) {
      const cur = regPts[s0];
      if (eyeColor(board, n, (cur / n) | 0, cur % n) === 0) continue;
      e = 1;
    } else if (regSize[rid] <= 5) {
      e = 1;
    } else {
      e = 2;
    }
    const roots = new Set();
    for (let i = s0; i < e0; i++) {
      const cur = regPts[i];
      const cr = (cur / n) | 0;
      const cc = cur % n;
      for (let k = 0; k < 4; k++) {
        const nr = cr + DR[k];
        const nc = cc + DC[k];
        if (!inBoard(nr, nc, n)) continue;
        const ni = nr * n + nc;
        if (board[ni] === col) roots.add(findRoot(parent, labels[ni]));
      }
    }
    for (const g of roots) grpEye[g] += e;
  }

  for (let idx = 0; idx < nn; idx++) {
    const b = board[idx];
    if (b === 0) continue;
    const gid = labels[idx];
    const root = findRoot(parent, gid);
    const urg = (4 - Math.min(lib[gid], 4)) / 3;
    const ee = grpEye[root];
    const eye = ee === 0 ? 0 : (ee === 1 ? 0.5 : 1);
    const off = b === player ? base : base + nn;
    out[off + idx] = urg;
    const eyeOff = b === player ? base + 3 * nn : base + 4 * nn;
    out[eyeOff + idx] = eye;
  }
  return out;
}


export class GoGame {
  constructor(boardSize = 19, komi = KOMI) {
    this.boardSize = boardSize;
    this.komi = komi;
    this.passIndex = boardSize * boardSize;
    const rules = BOARD_RULES[boardSize] || BOARD_RULES[19];
    this.maxMoves = rules.maxMoves;
    this.minMoves = rules.minMoves;
    this.passLimit = rules.passLimit;
    this.reset();
  }

  reset() {
    const n = this.boardSize;
    this.board = new Int8Array(n * n);
    this.prevBoard = null;
    this.koPoint = -1;
    this.currentPlayer = 1;
    this.lastMove = -2;
    this.moveCount = 0;
    this.consecutivePasses = 0;
    this.blackPasses = 0;
    this.whitePasses = 0;
    this.gameOver = false;
    this.winner = null;
    this.finalPoints = 0;
  }

  copy() {
    const g = Object.create(GoGame.prototype);
    g.boardSize = this.boardSize;
    g.komi = this.komi;
    g.passIndex = this.passIndex;
    g.maxMoves = this.maxMoves;
    g.minMoves = this.minMoves;
    g.passLimit = this.passLimit;
    g.board = this.board.slice();
    g.prevBoard = this.prevBoard ? this.prevBoard.slice() : null;
    g.koPoint = this.koPoint;
    g.currentPlayer = this.currentPlayer;
    g.lastMove = this.lastMove;
    g.moveCount = this.moveCount;
    g.consecutivePasses = this.consecutivePasses;
    g.blackPasses = this.blackPasses;
    g.whitePasses = this.whitePasses;
    g.gameOver = this.gameOver;
    g.winner = this.winner;
    g.finalPoints = this.finalPoints;
    return g;
  }

  forcedPass() {
    return this.currentPlayer === 1
      ? this.blackPasses >= this.passLimit
      : this.whitePasses >= this.passLimit;
  }

  passLegal() {
    return !this.gameOver && this.moveCount >= this.minMoves;
  }

  isPass(move) {
    return move === this.passIndex || move === PASS_MOVE;
  }

  isLegal(move) {
    if (this.gameOver) return false;
    if (this.isPass(move)) return this.passLegal();
    if (this.forcedPass()) return false;
    if (move < 0 || move >= this.passIndex || this.board[move] !== 0) return false;
    const r = (move / this.boardSize) | 0;
    const c = move % this.boardSize;
    let hasAir = false;
    for (let k = 0; k < 4; k++) {
      const nr = r + DR[k];
      const nc = c + DC[k];
      if (inBoard(nr, nc, this.boardSize) && this.board[nr * this.boardSize + nc] === 0) {
        hasAir = true;
        break;
      }
    }
    if (hasAir && move !== this.koPoint) return true;
    const copy = this.board.slice();
    return applyMoveRaw(copy, this.boardSize, move, this.currentPlayer, this.prevBoard) !== null;
  }

  getLegalMoves() {
    if (this.gameOver) return [];
    if (this.forcedPass()) return [this.passIndex];
    const n = this.boardSize;
    const out = [];
    for (let idx = 0; idx < this.passIndex; idx++) {
      if (this.board[idx] !== 0) continue;
      const r = (idx / n) | 0;
      const c = idx % n;
      let hasAir = false;
      for (let k = 0; k < 4; k++) {
        const nr = r + DR[k];
        const nc = c + DC[k];
        if (inBoard(nr, nc, n) && this.board[nr * n + nc] === 0) {
          hasAir = true;
          break;
        }
      }
      if (hasAir && idx !== this.koPoint) {
        out.push(idx);
      } else {
        const copy = this.board.slice();
        if (applyMoveRaw(copy, n, idx, this.currentPlayer, this.prevBoard) !== null) out.push(idx);
      }
    }
    if (this.moveCount >= this.minMoves) out.push(this.passIndex);
    return out;
  }

  getLegalMask() {
    const mask = new Float32Array(this.passIndex + 1);
    for (const m of this.getLegalMoves()) mask[m] = 1;
    return mask;
  }

  play(move) {
    if (this.gameOver) return false;
    if (this.isPass(move)) {
      if (!this.passLegal()) return false;
      if (this.currentPlayer === 1) this.blackPasses++;
      else this.whitePasses++;
      this.consecutivePasses++;
      this.moveCount++;
      this.lastMove = this.passIndex;
      this.koPoint = -1;
      this.prevBoard = this.board;
      this.currentPlayer = -this.currentPlayer;
      if (this.consecutivePasses >= 2 || this.moveCount >= this.maxMoves) this.finish();
      return true;
    }
    if (this.forcedPass()) return false;
    if (move < 0 || move >= this.passIndex || this.board[move] !== 0) return false;
    const before = this.board;
    const next = before.slice();
    const res = applyMoveRaw(next, this.boardSize, move, this.currentPlayer, this.prevBoard);
    if (!res) return false;
    const player = this.currentPlayer;
    this.board = next;
    this.prevBoard = before;
    this.koPoint = calcKoPoint(next, this.boardSize, move, res.captured, player);
    this.lastMove = move;
    this.consecutivePasses = 0;
    this.currentPlayer = -this.currentPlayer;
    this.moveCount++;
    if (this.moveCount >= this.maxMoves) this.finish();
    return true;
  }

  finish() {
    const s = this.score();
    this.finalPoints = s.finalDiff;
    this.winner = s.finalDiff > 0 ? 1 : (s.finalDiff < 0 ? -1 : 0);
    this.gameOver = true;
  }

  deadLumpMask() {
    const n = this.boardSize;
    const nn = n * n;
    const labels = new Int32Array(nn);
    const libSeen = new Int32Array(nn);
    const lib = new Int32Array(nn + 1);
    const size = new Int32Array(nn + 1);
    const stack = new Int32Array(nn);
    let gid = 0;
    for (let idx = 0; idx < nn; idx++) {
      if (this.board[idx] === 0 || labels[idx] !== 0) continue;
      const g = ++gid;
      let top = 0;
      stack[top++] = idx;
      labels[idx] = g;
      while (top > 0) {
        const cur = stack[--top];
        size[g]++;
        const r = (cur / n) | 0;
        const c = cur % n;
        for (let k = 0; k < 4; k++) {
          const nr = r + DR[k];
          const nc = c + DC[k];
          if (!inBoard(nr, nc, n)) continue;
          const ni = nr * n + nc;
          if (this.board[ni] === 0) {
            if (libSeen[ni] !== g) {
              libSeen[ni] = g;
              lib[g]++;
            }
          } else if (this.board[ni] === this.board[cur] && labels[ni] === 0) {
            labels[ni] = g;
            stack[top++] = ni;
          }
        }
      }
    }
    const dead = new Uint8Array(nn);
    for (let idx = 0; idx < nn; idx++) {
      if (this.board[idx] === 0) continue;
      const g = labels[idx];
      if (size[g] <= 2 || lib[g] <= 1) dead[idx] = 1;
    }
    return dead;
  }

  score() {
    const n = this.boardSize;
    const clean = this.board.slice();
    const dead = this.deadLumpMask();
    for (let i = 0; i < clean.length; i++) if (dead[i]) clean[i] = 0;
    let blackStones = 0;
    let whiteStones = 0;
    for (let i = 0; i < clean.length; i++) {
      if (clean[i] === 1) blackStones++;
      else if (clean[i] === -1) whiteStones++;
    }
    const seen = new Uint8Array(n * n);
    const stack = new Int32Array(n * n);
    let tb = 0;
    let tw = 0;
    let neutral = 0;
    for (let idx = 0; idx < n * n; idx++) {
      if (clean[idx] !== 0 || seen[idx]) continue;
      let top = 0;
      stack[top++] = idx;
      seen[idx] = 1;
      let size = 0;
      let hasB = false;
      let hasW = false;
      while (top > 0) {
        const cur = stack[--top];
        size++;
        const r = (cur / n) | 0;
        const c = cur % n;
        for (let k = 0; k < 4; k++) {
          const nr = r + DR[k];
          const nc = c + DC[k];
          if (!inBoard(nr, nc, n)) continue;
          const ni = nr * n + nc;
          if (clean[ni] === 1) hasB = true;
          else if (clean[ni] === -1) hasW = true;
          else if (!seen[ni]) {
            seen[ni] = 1;
            stack[top++] = ni;
          }
        }
      }
      if (hasB && !hasW) tb += size;
      else if (hasW && !hasB) tw += size;
      else neutral += size;
    }
    const finalDiff = (blackStones + tb) - (whiteStones + tw + this.komi);
    return { blackStones, whiteStones, tb, tw, neutral, finalDiff };
  }

  canonicalState(legalMask = null) {
    const n = this.boardSize;
    const nn = n * n;
    const out = new Float32Array(INPUT_CHANNELS * nn);
    const player = this.currentPlayer;
    const kch = player * this.komi / 14;
    // 边界通道
    for (let c = 0; c < n; c++) {
      out[CH_BOUNDARY * nn + c] = 1;
      out[CH_BOUNDARY * nn + (n - 1) * n + c] = 1;
    }
    for (let r = 0; r < n; r++) {
      out[CH_BOUNDARY * nn + r * n] = 1;
      out[CH_BOUNDARY * nn + r * n + n - 1] = 1;
    }
    const last = this.lastMove >= 0 && this.lastMove < nn ? this.lastMove : -1;
    for (let idx = 0; idx < nn; idx++) {
      const v = this.board[idx];
      if (v === player) out[CH_OWN_STONES * nn + idx] = 1;
      else if (v === -player) out[CH_OPP_STONES * nn + idx] = 1;
      if (idx === last) out[CH_LAST_MOVE * nn + idx] = 1;
      out[CH_KOMI * nn + idx] = kch;
    }
    if (legalMask) {
      for (let idx = 0; idx < nn; idx++) out[CH_LEGAL_MASK * nn + idx] = legalMask[idx];
    }
    groupChannels(this.board, n, player, out, CH_OWN_URGENCY * nn);
    return out;
  }

  // 仅用于调试/测试。
  boardString() {
    return Array.from(this.board).join(',');
  }
}

export function moveToRC(move, n) {
  if (move < 0 || move >= n * n) return null;
  return { r: (move / n) | 0, c: move % n };
}
