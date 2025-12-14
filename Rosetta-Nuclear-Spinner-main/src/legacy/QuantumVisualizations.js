// ================================================================
// QUANTUM VISUALIZATIONS
// Helper components for rendering QuantumAPL state
// ================================================================

class DensityMatrixViz {
    render(rho, canvas, options = {}) {
        if (!rho || !canvas || typeof canvas.getContext !== 'function') return;
        const ctx = canvas.getContext('2d');
        const dim = rho.rows || 0;
        if (!dim) return;

        const cellSize = canvas.width / dim;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                const element = rho.get(i, j);
                const amp = Math.min(1, element.abs());
                const phase = Math.atan2(element.im, element.re);
                const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
                const lightness = (options.minBrightness || 30) + amp * (options.maxBrightness || 60);

                ctx.fillStyle = `hsl(${hue}, 100%, ${Math.min(100, lightness)}%)`;
                ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
            }
        }

        if (options.highlightDiagonal !== false) {
            ctx.strokeStyle = options.diagonalColor || '#ffffffaa';
            ctx.lineWidth = options.diagonalWidth || 2;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
            for (let k = 0; k < dim; k++) {
                ctx.strokeRect(k * cellSize, k * cellSize, cellSize, cellSize);
            }
        }
    }
}

class TruthBlochSphere {
    constructor(threeRef) {
        this.THREE = threeRef || (typeof THREE !== 'undefined' ? THREE : null);
        this.currentArrow = null;
        this.criticalRing = null;
    }

    render(truthState, scene) {
        if (!this.THREE || !scene || !truthState) return null;
        const { TRUE = 0, UNTRUE = 0, PARADOX = 0 } = truthState;
        const total = TRUE + UNTRUE + PARADOX || 1;
        const normalized = {
            TRUE: TRUE / total,
            UNTRUE: UNTRUE / total,
            PARADOX: PARADOX / total
        };

        const theta = Math.acos(Math.min(1, Math.max(-1, normalized.TRUE - normalized.UNTRUE)));
        const phi = Math.atan2(normalized.PARADOX || 0.0001, (normalized.TRUE + normalized.UNTRUE) || 0.0001);

        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);

        if (this.currentArrow) {
            scene.remove(this.currentArrow);
        }

        const arrow = new this.THREE.ArrowHelper(
            new this.THREE.Vector3(x, y, z).normalize(),
            new this.THREE.Vector3(0, 0, 0),
            1,
            0x00ff88
        );
        scene.add(arrow);
        this.currentArrow = arrow;

        if (!this.criticalRing) {
            this.drawCriticalRing(scene, Math.sqrt(3) / 2);
        }

        return arrow;
    }

    drawCriticalRing(scene, radius) {
        if (!this.THREE || !scene) return null;
        if (this.criticalRing) {
            scene.remove(this.criticalRing);
        }

        const geometry = new this.THREE.RingGeometry(radius - 0.01, radius + 0.01, 64);
        const material = new this.THREE.MeshBasicMaterial({ color: 0xffaa00, side: this.THREE.DoubleSide });
        const mesh = new this.THREE.Mesh(geometry, material);
        mesh.rotation.x = Math.PI / 2;
        scene.add(mesh);
        this.criticalRing = mesh;
        return mesh;
    }
}

class CoherenceGraph {
    constructor(getCoherencesFn) {
        this.getCoherences = getCoherencesFn;
    }

    render(quantumOrRho, graph, threshold = 0.01) {
        if (!graph) return;
        const coherences = typeof this.getCoherences === 'function'
            ? this.getCoherences()
            : quantumOrRho?.getCoherences?.() || [];

        const rho = quantumOrRho?.rho || quantumOrRho;
        if (!rho) return;

        coherences.forEach(({ i, j, value }) => {
            if (value < threshold) return;
            const element = rho.get(i, j);
            const phase = Math.atan2(element.im, element.re);
            const color = this.phaseToColor(phase);
            const payload = {
                thickness: value * 5,
                color,
                phase,
                magnitude: value
            };

            if (typeof graph.addEdge === 'function') {
                graph.addEdge(i, j, payload);
            } else {
                if (!graph.edges) graph.edges = [];
                graph.edges.push({ from: i, to: j, ...payload });
            }
        });
    }

    phaseToColor(phase) {
        const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
        return `hsl(${hue}, 90%, 55%)`;
    }

    renderToCanvas(rho, canvas, options = {}) {
        if (!canvas || !rho) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        const nodeCount = options.nodeCount || Math.min(rho.rows, options.maxNodes || 12);
        const populations = [];
        for (let i = 0; i < rho.rows; i++) {
            populations.push({ index: i, value: Math.max(0, rho.get(i, i).re) });
        }
        const nodes = populations
            .sort((a, b) => b.value - a.value)
            .slice(0, nodeCount);

        const radius = Math.min(width, height) * 0.35;
        const center = { x: width / 2, y: height / 2 };
        const positions = new Map();

        nodes.forEach((node, idx) => {
            const angle = (idx / nodes.length) * Math.PI * 2;
            positions.set(node.index, {
                x: center.x + radius * Math.cos(angle),
                y: center.y + radius * Math.sin(angle),
                value: node.value
            });
        });

        const coherences = typeof this.getCoherences === 'function'
            ? this.getCoherences()
            : rho.getCoherences?.() || [];

        ctx.lineWidth = 1.5;
        coherences.forEach(({ i, j, value }) => {
            if (!positions.has(i) || !positions.has(j)) return;
            if (value < (options.threshold || 0.01)) return;
            const element = rho.get(i, j);
            ctx.strokeStyle = this.phaseToColor(Math.atan2(element.im, element.re));
            ctx.lineWidth = Math.max(1, value * 8);
            const pi = positions.get(i);
            const pj = positions.get(j);
            ctx.beginPath();
            ctx.moveTo(pi.x, pi.y);
            ctx.lineTo(pj.x, pj.y);
            ctx.stroke();
        });

        nodes.forEach(node => {
            const pos = positions.get(node.index);
            ctx.fillStyle = `rgba(255,255,255,${0.5 + 0.5 * pos.value})`;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 6 + pos.value * 10, 0, Math.PI * 2);
            ctx.fill();
        });
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DensityMatrixViz, TruthBlochSphere, CoherenceGraph };
}
