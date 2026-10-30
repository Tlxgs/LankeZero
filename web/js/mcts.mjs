// 浏览器端 MCTS：与 python/mcts.py 同一套选择、候选裁剪、虚拟损失和根 pass 探测。
// 评估函数由 worker 注入：async (states: Float32Array[]) => {policies, values, wins}

export const PASS_MOVE = -1;

export class Node {
  constructor(parent, move, prior = 0) {
    this.parent = parent;
    this.move = move;
    this.prior = prior;
    this.children = null;
    this.visits = 0;
    this.vsum = 0;
    this.vloss = 0;
    this.expanded = false;
  }
}

function utility(winLogit, score, alpha, scale = 30) {
  return (1 - alpha) * Math.tanh(winLogit) + alpha * Math.tanh(score / scale);
}

export class MCTS {
  constructor(options = {}) {
    this.evaluate = options.evaluate;
    this.boardSize = options.boardSize || 19;
    this.cPuct = options.cPuct ?? 1.0;
    this.cPuctLog = options.cPuctLog ?? 0.45;
    this.cPuctBase = options.cPuctBase ?? 500;
    this.utilStdevPrior = options.utilStdevPrior ?? 0.05;
    this.utilStdevScale = options.utilStdevScale ?? 0.85;
    this.fpuReduction = options.fpuReduction ?? 0.05;
    this.virtualLoss = options.virtualLoss ?? 5;
    this.batchSize = Math.max(1, options.batchSize || 16);
    this.candidateMass = options.candidateMass ?? 0.95;
    this.minCandidates = options.minCandidates ?? 8;
    this.alpha = options.alpha ?? 0.5;
    this.temperature = options.temperature ?? 0;
    this.numSimulations = options.numSimulations ?? 300;
    this.root = null;
  }

  reset() {
    this.root = null;
  }

  async initRoot(game) {
    const legal = game.getLegalMoves();
    const mask = new Float32Array(game.passIndex + 1);
    for (const m of legal) mask[m] = 1;
    const state = game.canonicalState(mask);
    const ev = await this.evaluate([state], [mask]);
    const policy = ev.policies[0];
    const root = new Node(null, -1, 0);
    this.root = root;
    root.visits = 0;
    this._expand(root, legal, policy, true);
    return root;
  }

  async updateRoot(game, move) {
    if (!this.root) return this.initRoot(game);
    const found = this.root.children ? this.root.children.find((c) => c.move === move) : null;
    if (found) {
      this.root = found;
      this.root.parent = null;
      this.root.vloss = 0;
      return this.root;
    }
    return this.initRoot(game);
  }

  async ensureRoot(game) {
    if (!this.root) return this.initRoot(game);
    if (this.root.expanded) return this.root;
    // 人类可能走了上一棵树里“有先验但 0 访问”的候选点，updateRoot 会把它
    // 提升为根；此时根还没有展开。必须补一次推理/展开，否则 search 会直接
    // 返回 pass，而开局 pass 又不合法。
    const legal = game.getLegalMoves();
    const mask = new Float32Array(game.passIndex + 1);
    for (const m of legal) mask[m] = 1;
    const ev = await this.evaluate([game.canonicalState(mask)], [mask]);
    this._expand(this.root, legal, ev.policies[0], false);
    return this.root;
  }

  _expand(node, legal, policy, isRoot) {
    if (node.expanded) return;
    node.expanded = true;
    if (!legal || legal.length === 0) {
      node.children = [];
      return;
    }
    const total = legal.reduce((s, m) => s + policy[m], 0);
    let probs;
    if (total <= 0) {
      probs = legal.map(() => 1 / legal.length);
    } else {
      probs = legal.map((m) => policy[m] / total);
    }
    const order = legal.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b);
    const target = total > 0 ? total * this.candidateMass : 1;
    const minK = Math.min(this.minCandidates, legal.length);
    let k = 0;
    let cum = 0;
    while (k < legal.length && (k < minK || cum < target)) {
      cum += total > 0 ? policy[legal[order[k]]] : 1;
      k++;
    }
    if (k <= 0) {
      k = 1;
      cum = total > 0 ? policy[legal[order[0]]] : 1;
    }
    let chosen = [];
    for (let i = 0; i < k; i++) chosen.push(order[i]);
    if (isRoot) {
      const pass = legal.indexOf(this.passIndex ?? this.boardSize * this.boardSize);
      if (pass >= 0 && !chosen.includes(pass)) chosen.push(pass);
    }
    let denom = 0;
    for (const i of chosen) denom += total > 0 ? policy[legal[i]] : 1;
    if (denom <= 0) denom = chosen.length;
    node.children = chosen.map((i) => {
      const p = total > 0 ? policy[legal[i]] : 1;
      return new Node(node, legal[i], p / denom);
    });
  }

  _selectChild(node) {
    const children = node.children;
    const visits = node.visits;
    const pv = visits > 0 ? node.vsum / visits : 0;
    let qsum = 0;
    let qsq = 0;
    let nvis = 0;
    for (const ch of children) {
      const vis = ch.visits;
      const q = vis > 0 ? -ch.vsum / vis : pv - this.fpuReduction;
      if (vis > 0) {
        qsum += q;
        qsq += q * q;
        nvis++;
      }
    }
    let qvar;
    if (nvis >= 2) {
      const qmean = qsum / nvis;
      qvar = qsq / nvis - qmean * qmean;
      if (qvar < 0) qvar = 0;
    } else {
      qvar = this.utilStdevPrior * this.utilStdevPrior;
    }
    let exploreC = this.cPuct + this.cPuctLog * Math.log((visits + this.cPuctBase) / this.cPuctBase);
    exploreC *= 1 + this.utilStdevScale * (Math.sqrt(qvar) / this.utilStdevPrior - 1);
    if (exploreC < 0) exploreC = 0;
    const sqrtN = Math.sqrt(1 + visits + node.vloss);
    let best = 0;
    let bestScore = -1e18;
    for (let i = 0; i < children.length; i++) {
      const ch = children[i];
      const vis = ch.visits;
      const q = vis > 0 ? -ch.vsum / vis : pv - this.fpuReduction;
      const score = q + exploreC * ch.prior * sqrtN / (1 + vis + ch.vloss);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  _select(game) {
    let node = this.root;
    const path = [node];
    node.vloss += this.virtualLoss;
    while (node.children && node.children.length) {
      const child = node.children[this._selectChild(node)];
      if (!game.play(child.move)) {
        throw new Error('MCTS selected an illegal move');
      }
      child.vloss += this.virtualLoss;
      path.push(child);
      node = child;
      if (game.gameOver) break;
    }
    return { node, path };
  }

  _backup(path, value) {
    for (let i = path.length - 1; i >= 0; i--) {
      const node = path[i];
      node.vloss -= this.virtualLoss;
      node.visits++;
      node.vsum += value;
      value = -value;
    }
  }

  async _runBatch(game, count, isCancelled) {
    const sims = [];
    for (let i = 0; i < count; i++) {
      if (isCancelled && isCancelled()) return false;
      const gc = game.copy();
      const sel = this._select(gc);
      const sim = { ...sel, game: gc, legal: null, mask: null, policy: null };
      if (!gc.gameOver) {
        sim.legal = gc.getLegalMoves();
        sim.mask = new Float32Array(gc.passIndex + 1);
        for (const m of sim.legal) sim.mask[m] = 1;
      }
      sims.push(sim);
    }
    if (isCancelled && isCancelled()) return false;
    const pending = sims.filter((s) => !s.game.gameOver);
    let evals = null;
    if (pending.length) {
      const states = pending.map((s) => s.game.canonicalState(s.mask));
      evals = await this.evaluate(states, pending.map((s) => s.mask));
      if (isCancelled && isCancelled()) return false;
    }
    const bySim = new Map();
    if (evals) for (let i = 0; i < pending.length; i++) bySim.set(pending[i], i);
    for (const sim of sims) {
      let value;
      if (sim.game.gameOver) {
        if (sim.game.winner === 0) value = 0;
        else {
          const termScore = sim.game.finalPoints * sim.game.currentPlayer;
          const termWl = sim.game.winner === sim.game.currentPlayer ? 6 : -6;
          value = utility(termWl, termScore, this.alpha);
        }
      } else {
        const i = bySim.get(sim);
        const rawValue = evals.values[i];
        const win = evals.wins[i];
        sim.policy = evals.policies[i];
        const score = rawValue - sim.game.komi * sim.game.currentPlayer;
        value = utility(win, score, this.alpha);
      }
      this._backup(sim.path, value);
      if (!sim.game.gameOver && sim.policy && !sim.node.expanded) {
        this._expand(sim.node, sim.legal, sim.policy, false);
      }
    }
    return true;
  }

  async probePass(game, isCancelled) {
    if (!this.root || !this.root.children) return;
    const passIdx = this.passIndex ?? this.boardSize * this.boardSize;
    const child = this.root.children.find((c) => c.move === passIdx);
    if (!child || child.visits > 0) return;
    const saved = this.root.children.map((c) => c.prior);
    for (const c of this.root.children) c.prior = 0;
    child.prior = 1;
    try {
      await this._runBatch(game, 1, isCancelled);
    } finally {
      this.root.children.forEach((c, i) => { c.prior = saved[i]; });
    }
  }

  result() {
    if (!this.root || !this.root.children || !this.root.children.length) {
      return { move: this.passIndex ?? this.boardSize * this.boardSize, visits: 0, policy: {} };
    }
    let best = this.root.children[0];
    for (const c of this.root.children) if (c.visits > best.visits) best = c;
    if (best.visits === 0) {
      for (const c of this.root.children) if (c.prior > best.prior) best = c;
    }
    const policy = {};
    for (const c of this.root.children) policy[c.move] = c.visits;
    return { move: best.move, visits: best.visits, policy };
  }

  async search(game, targetVisits = this.numSimulations, onProgress = null, isCancelled = null) {
    await this.ensureRoot(game);
    if (!this.root || !this.root.children || !this.root.children.length) return this.result();
    await this.probePass(game, isCancelled);
    while (this.root.visits < targetVisits) {
      if (isCancelled && isCancelled()) return null;
      const batch = Math.min(this.batchSize, targetVisits - this.root.visits);
      const ok = await this._runBatch(game, batch, isCancelled);
      if (!ok) return null;
      if (onProgress) onProgress(this.root.visits, targetVisits);
    }
    return this.result();
  }
}
