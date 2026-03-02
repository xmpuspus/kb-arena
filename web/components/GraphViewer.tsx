"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, string>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (node: GraphNode) => void;
}

const TYPE_COLORS: Record<string, { fill: string; glow: string }> = {
  // AWS node types
  Service:   { fill: "#2563eb", glow: "#3b82f6" },
  Resource:  { fill: "#059669", glow: "#10b981" },
  Policy:    { fill: "#d97706", glow: "#f59e0b" },
  Feature:   { fill: "#7c3aed", glow: "#8b5cf6" },
  // Generic node types
  Module:    { fill: "#2563eb", glow: "#3b82f6" },
  Class:     { fill: "#059669", glow: "#10b981" },
  Function:  { fill: "#d97706", glow: "#f59e0b" },
  Method:    { fill: "#7c3aed", glow: "#8b5cf6" },
  Variable:  { fill: "#dc2626", glow: "#ef4444" },
  Exception: { fill: "#db2777", glow: "#ec4899" },
  Constant:  { fill: "#0891b2", glow: "#06b6d4" },
  Parameter: { fill: "#65a30d", glow: "#84cc16" },
  Default:   { fill: "#475569", glow: "#64748b" },
};

function getColor(type: string) {
  return TYPE_COLORS[type] ?? TYPE_COLORS.Default;
}

// Simulation constants
const BASE_RADIUS = 6;
const DEGREE_SCALE = 4;
const REPULSION = 400;
const LINK_STRENGTH = 0.2;
const IDEAL_DIST = 160;
const CENTER_STRENGTH = 0.008;
const COLLISION_PAD = 4;
const ALPHA_MIN = 0.001;
const ALPHA_DECAY = 0.0228;
const VELOCITY_DECAY = 0.6;

function nodeRadius(degree: number) {
  return BASE_RADIUS + Math.log2(1 + degree) * DEGREE_SCALE;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

export default function GraphViewer({ nodes, edges, onNodeClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);

  // Layout refs
  const posRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const velRef = useRef<Map<string, { vx: number; vy: number }>>(new Map());
  const animRef = useRef<number>(0);

  // Interaction refs
  const dragging = useRef<{ id: string } | null>(null);
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0 });
  const transform = useRef({ scale: 1, tx: 0, ty: 0 });
  const hoveredId = useRef<string | null>(null);
  const mouseScreen = useRef({ x: 0, y: 0 });

  // Animation refs
  const simState = useRef({ alpha: 1, stabilized: false });
  const dirty = useRef(true);
  const pulsePhase = useRef(0);
  const entryStart = useRef(0);

  // Pre-compute adjacency data
  const { degreeMap, neighborMap, edgeGroups } = useMemo(() => {
    const deg = new Map<string, number>();
    const neigh = new Map<string, Set<string>>();
    const groups = new Map<string, GraphEdge[]>();

    for (const n of nodes) {
      deg.set(n.id, 0);
      neigh.set(n.id, new Set());
    }
    for (const e of edges) {
      deg.set(e.source, (deg.get(e.source) ?? 0) + 1);
      deg.set(e.target, (deg.get(e.target) ?? 0) + 1);
      neigh.get(e.source)?.add(e.target);
      neigh.get(e.target)?.add(e.source);
      const key = [e.source, e.target].sort().join("|");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(e);
    }
    return { degreeMap: deg, neighborMap: neigh, edgeGroups: groups };
  }, [nodes, edges]);

  // Node map for quick lookup
  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  // Search filter
  const visibleIds = useMemo(() => {
    if (!search) return new Set(nodes.map((n) => n.id));
    return new Set(
      nodes.filter((n) => n.label.toLowerCase().includes(search.toLowerCase())).map((n) => n.id),
    );
  }, [nodes, search]);

  // Initialize positions in circle
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    const cx = W / 2;
    const cy = H / 2;
    const r = Math.min(W, H) * 0.35;

    for (let i = 0; i < nodes.length; i++) {
      const angle = (2 * Math.PI * i) / nodes.length;
      posRef.current.set(nodes[i].id, {
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
      });
      velRef.current.set(nodes[i].id, { vx: 0, vy: 0 });
    }

    simState.current = { alpha: 1, stabilized: false };
    entryStart.current = performance.now();
    dirty.current = true;
  }, [nodes]);

  // Hit test: screen coords → node
  const hitTest = useCallback((sx: number, sy: number): GraphNode | null => {
    const { scale, tx, ty } = transform.current;
    const wx = (sx - tx) / scale;
    const wy = (sy - ty) / scale;
    let closest: GraphNode | null = null;
    let closestDist = Infinity;
    for (const node of nodes) {
      const p = posRef.current.get(node.id);
      if (!p) continue;
      const r = nodeRadius(degreeMap.get(node.id) ?? 0);
      const d = Math.sqrt((wx - p.x) ** 2 + (wy - p.y) ** 2);
      if (d < r + 6 && d < closestDist) {
        closest = node;
        closestDist = d;
      }
    }
    return closest;
  }, [nodes, degreeMap]);

  // Force simulation + render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let running = true;

    function applyForces() {
      const W = canvas!.width;
      const H = canvas!.height;
      const centerX = W / 2;
      const centerY = H / 2;
      const sim = simState.current;

      // Many-body repulsion
      for (const a of nodes) {
        const pa = posRef.current.get(a.id)!;
        const va = velRef.current.get(a.id)!;
        for (const b of nodes) {
          if (a.id >= b.id) continue;
          const pb = posRef.current.get(b.id)!;
          const vb = velRef.current.get(b.id)!;
          const dx = pa.x - pb.x;
          const dy = pa.y - pb.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = REPULSION * sim.alpha / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          va.vx += fx;
          va.vy += fy;
          vb.vx -= fx;
          vb.vy -= fy;
        }
      }

      // Link attraction
      for (const edge of edges) {
        const ps = posRef.current.get(edge.source);
        const pt = posRef.current.get(edge.target);
        const vs = velRef.current.get(edge.source);
        const vt = velRef.current.get(edge.target);
        if (!ps || !pt || !vs || !vt) continue;
        const dx = pt.x - ps.x;
        const dy = pt.y - ps.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const strength = (dist - IDEAL_DIST) / dist * LINK_STRENGTH * sim.alpha;
        vs.vx += dx * strength;
        vs.vy += dy * strength;
        vt.vx -= dx * strength;
        vt.vy -= dy * strength;
      }

      // Center gravity
      for (const node of nodes) {
        const p = posRef.current.get(node.id)!;
        const v = velRef.current.get(node.id)!;
        v.vx += (centerX - p.x) * CENTER_STRENGTH * sim.alpha;
        v.vy += (centerY - p.y) * CENTER_STRENGTH * sim.alpha;
      }

      // Collision detection
      for (const a of nodes) {
        const pa = posRef.current.get(a.id)!;
        const ra = nodeRadius(degreeMap.get(a.id) ?? 0);
        for (const b of nodes) {
          if (a.id >= b.id) continue;
          const pb = posRef.current.get(b.id)!;
          const rb = nodeRadius(degreeMap.get(b.id) ?? 0);
          const dx = pa.x - pb.x;
          const dy = pa.y - pb.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const minDist = ra + rb + COLLISION_PAD;
          if (dist < minDist) {
            const overlap = (minDist - dist) / dist * 0.5;
            pa.x += dx * overlap;
            pa.y += dy * overlap;
            pb.x -= dx * overlap;
            pb.y -= dy * overlap;
          }
        }
      }

      // Integrate velocities
      for (const node of nodes) {
        if (dragging.current?.id === node.id) continue;
        const p = posRef.current.get(node.id)!;
        const v = velRef.current.get(node.id)!;
        v.vx *= VELOCITY_DECAY;
        v.vy *= VELOCITY_DECAY;
        p.x += v.vx;
        p.y += v.vy;
      }

      // Alpha decay
      sim.alpha += (0 - sim.alpha) * ALPHA_DECAY;
      if (sim.alpha < ALPHA_MIN) {
        sim.stabilized = true;
      }
    }

    function draw(W: number, H: number) {
      const zoom = transform.current.scale;
      const { tx, ty } = transform.current;
      const hId = hoveredId.current;
      const hNeighbors = hId ? neighborMap.get(hId) : null;
      const isHoverActive = hId !== null;

      // Entry animation
      const elapsed = performance.now() - entryStart.current;
      const entryT = Math.min(1, elapsed / 800);
      const entry = entryT * entryT * (3 - 2 * entryT); // smoothstep

      // Layer 1: Background
      ctx!.clearRect(0, 0, W, H);
      ctx!.fillStyle = "#f8fafc";
      ctx!.fillRect(0, 0, W, H);

      ctx!.save();
      ctx!.translate(tx, ty);
      ctx!.scale(zoom, zoom);

      // Layer 2: Edges (bezier curves)
      for (const edge of edges) {
        const ps = posRef.current.get(edge.source);
        const pt = posRef.current.get(edge.target);
        if (!ps || !pt) continue;

        const srcNode = nodeMap.get(edge.source);
        const color = srcNode ? getColor(srcNode.type) : getColor("Default");

        const searchDim = search && !visibleIds.has(edge.source) && !visibleIds.has(edge.target);
        const isNeighborEdge = isHoverActive && (
          (edge.source === hId || edge.target === hId)
        );

        // Parallel edge offset
        const key = [edge.source, edge.target].sort().join("|");
        const group = edgeGroups.get(key) ?? [edge];
        const idx = group.indexOf(edge);
        const count = group.length;
        const offsetMul = count > 1 ? (idx - (count - 1) / 2) * 25 : 0;

        // Bezier control point
        const mx = (ps.x + pt.x) / 2;
        const my = (ps.y + pt.y) / 2;
        const dx = pt.x - ps.x;
        const dy = pt.y - ps.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = -dy / len;
        const ny = dx / len;
        const curveOffset = len * 0.12 + offsetMul;
        const cx = mx + nx * curveOffset;
        const cy = my + ny * curveOffset;

        ctx!.beginPath();
        ctx!.moveTo(ps.x, ps.y);
        ctx!.quadraticCurveTo(cx, cy, pt.x, pt.y);

        let alpha: string;
        let lineW: number;
        if (searchDim) {
          alpha = "14";
          lineW = 0.5;
        } else if (isHoverActive) {
          alpha = isNeighborEdge ? "80" : "0D";
          lineW = isNeighborEdge ? 2 : 0.5;
        } else {
          alpha = "50";
          lineW = 1.2;
        }

        ctx!.strokeStyle = color.fill + alpha;
        ctx!.lineWidth = lineW;
        ctx!.stroke();
      }

      // Layer 3: Edge labels (zoom > 1.0)
      if (zoom > 1.0) {
        for (const edge of edges) {
          const ps = posRef.current.get(edge.source);
          const pt = posRef.current.get(edge.target);
          if (!ps || !pt) continue;
          if (search && !visibleIds.has(edge.source) && !visibleIds.has(edge.target)) continue;
          if (isHoverActive && edge.source !== hId && edge.target !== hId) continue;

          const key = [edge.source, edge.target].sort().join("|");
          const group = edgeGroups.get(key) ?? [edge];
          const idx = group.indexOf(edge);
          const count = group.length;
          const offsetMul = count > 1 ? (idx - (count - 1) / 2) * 25 : 0;

          const mx = (ps.x + pt.x) / 2;
          const my = (ps.y + pt.y) / 2;
          const dx = pt.x - ps.x;
          const dy = pt.y - ps.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const nx = -dy / len;
          const ny = dx / len;
          const curveOff = len * 0.12 + offsetMul;
          const bcx = mx + nx * curveOff;
          const bcy = my + ny * curveOff;

          // Point on bezier at t=0.5
          const t = 0.5;
          const labelX = (1-t)*(1-t)*ps.x + 2*(1-t)*t*bcx + t*t*pt.x;
          const labelY = (1-t)*(1-t)*ps.y + 2*(1-t)*t*bcy + t*t*pt.y;

          let angle = Math.atan2(dy, dx);
          if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI;

          ctx!.save();
          ctx!.translate(labelX, labelY);
          ctx!.rotate(angle);
          ctx!.font = `${Math.max(8, 9 / zoom)}px -apple-system, system-ui, sans-serif`;
          ctx!.fillStyle = "#94a3b8";
          ctx!.textAlign = "center";
          ctx!.textBaseline = "bottom";
          ctx!.fillText(edge.label, 0, -3);
          ctx!.restore();
        }
      }

      // Layer 4+5: Nodes (glow + fill)
      for (const node of nodes) {
        const p = posRef.current.get(node.id);
        if (!p) continue;
        const color = getColor(node.type);
        const degree = degreeMap.get(node.id) ?? 0;
        const r = nodeRadius(degree) * entry;
        const isSelected = selected?.id === node.id;
        const isHovered = hoveredId.current === node.id;
        const isNeighbor = isHoverActive && (hNeighbors?.has(node.id) || node.id === hId);
        const searchDim = search && !visibleIds.has(node.id);

        let nodeAlpha = entry;
        if (searchDim) nodeAlpha = 0.15;
        else if (isHoverActive && !isNeighbor) nodeAlpha = 0.15;

        ctx!.globalAlpha = nodeAlpha;

        // Glow
        ctx!.save();
        if (isSelected) {
          ctx!.shadowColor = color.glow;
          ctx!.shadowBlur = 20 + Math.sin(pulsePhase.current) * 5;
        } else if (isHovered) {
          ctx!.shadowColor = color.glow;
          ctx!.shadowBlur = 16;
        } else if (!searchDim && !(isHoverActive && !isNeighbor)) {
          ctx!.shadowColor = color.glow;
          ctx!.shadowBlur = 10;
        }
        ctx!.shadowOffsetX = 0;
        ctx!.shadowOffsetY = 0;

        ctx!.beginPath();
        ctx!.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx!.fillStyle = color.fill;
        ctx!.fill();
        ctx!.restore();

        // White inner highlight
        if (r > 4) {
          ctx!.beginPath();
          ctx!.arc(p.x - r * 0.2, p.y - r * 0.25, r * 0.35, 0, Math.PI * 2);
          ctx!.fillStyle = "rgba(255,255,255,0.25)";
          ctx!.fill();
        }

        // Selection ring
        if (isSelected) {
          ctx!.beginPath();
          ctx!.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
          ctx!.strokeStyle = color.fill;
          ctx!.lineWidth = 2;
          ctx!.stroke();
        }

        ctx!.globalAlpha = 1;
      }

      // Layer 6: Labels (zoom > 0.5)
      if (zoom > 0.5) {
        const fontSize = Math.max(10, Math.min(13, 11 / zoom));
        ctx!.font = `500 ${fontSize}px -apple-system, system-ui, sans-serif`;

        for (const node of nodes) {
          const p = posRef.current.get(node.id);
          if (!p) continue;
          const degree = degreeMap.get(node.id) ?? 0;
          const r = nodeRadius(degree) * entry;
          const searchDim = search && !visibleIds.has(node.id);
          const isNeighbor = isHoverActive && (hNeighbors?.has(node.id) || node.id === hId);

          let labelAlpha = entry;
          if (searchDim) labelAlpha = 0.12;
          else if (isHoverActive && !isNeighbor) labelAlpha = 0.12;

          ctx!.globalAlpha = labelAlpha;
          ctx!.fillStyle = "#334155";
          ctx!.textBaseline = "middle";
          const label = node.label.length > 18 ? node.label.slice(0, 16) + "\u2026" : node.label;
          ctx!.fillText(label, p.x + r + 5, p.y);
          ctx!.globalAlpha = 1;
        }
      }

      ctx!.restore();

      // Layer 7: Hover tooltip (screen space)
      if (hId && !searchDim(hId)) {
        const hNode = nodeMap.get(hId);
        if (hNode) {
          const hDeg = degreeMap.get(hId) ?? 0;
          const hColor = getColor(hNode.type);
          const tipX = mouseScreen.current.x + 14;
          const tipY = mouseScreen.current.y - 8;
          const tipW = 170;
          const tipH = 50;

          // Keep tooltip in bounds
          const adjustedX = tipX + tipW > W ? tipX - tipW - 28 : tipX;
          const adjustedY = tipY + tipH > H ? tipY - tipH : tipY;

          ctx!.fillStyle = "#ffffff";
          ctx!.strokeStyle = "#e2e8f0";
          ctx!.lineWidth = 1;
          ctx!.shadowColor = "rgba(0,0,0,0.08)";
          ctx!.shadowBlur = 8;
          ctx!.shadowOffsetX = 0;
          ctx!.shadowOffsetY = 2;
          roundRect(ctx!, adjustedX, adjustedY, tipW, tipH, 6);
          ctx!.fill();
          ctx!.shadowBlur = 0;
          ctx!.stroke();

          // Type dot
          ctx!.beginPath();
          ctx!.arc(adjustedX + 12, adjustedY + 17, 4, 0, Math.PI * 2);
          ctx!.fillStyle = hColor.fill;
          ctx!.fill();

          ctx!.fillStyle = "#0f172a";
          ctx!.font = "600 12px -apple-system, system-ui, sans-serif";
          ctx!.textBaseline = "middle";
          ctx!.fillText(hNode.label, adjustedX + 22, adjustedY + 17);

          ctx!.fillStyle = "#64748b";
          ctx!.font = "11px -apple-system, system-ui, sans-serif";
          ctx!.fillText(
            `${hNode.type}  \u00b7  ${hDeg} connection${hDeg !== 1 ? "s" : ""}`,
            adjustedX + 12,
            adjustedY + 36,
          );
        }
      }

      // Helper to check search dim inline
      function searchDim(id: string) {
        return search !== "" && !visibleIds.has(id);
      }
    }

    function loop() {
      if (!running) return;
      const sim = simState.current;

      if (!sim.stabilized) {
        applyForces();
        dirty.current = true;
      }

      // Pulse for selected node
      if (selected) {
        pulsePhase.current += 0.05;
        dirty.current = true;
      }

      // Entry animation still going
      if (performance.now() - entryStart.current < 850) {
        dirty.current = true;
      }

      if (dirty.current) {
        draw(canvas!.width, canvas!.height);
        dirty.current = false;
      }

      animRef.current = requestAnimationFrame(loop);
    }

    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; cancelAnimationFrame(animRef.current); };
  }, [nodes, edges, search, selected, degreeMap, neighborMap, edgeGroups, nodeMap, visibleIds, hitTest]);

  // --- Interaction handlers ---

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = hitTest(cx, cy);

    if (hit) {
      dragging.current = { id: hit.id };
      setSelected(hit);
      onNodeClick?.(hit);
      // Reheat simulation
      simState.current.alpha = 0.3;
      simState.current.stabilized = false;
      dirty.current = true;
    } else {
      isPanning.current = true;
      panStart.current = {
        x: cx - transform.current.tx,
        y: cy - transform.current.ty,
      };
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    mouseScreen.current = { x: cx, y: cy };

    if (dragging.current) {
      const { scale, tx, ty } = transform.current;
      const wx = (cx - tx) / scale;
      const wy = (cy - ty) / scale;
      posRef.current.set(dragging.current.id, { x: wx, y: wy });
      velRef.current.set(dragging.current.id, { vx: 0, vy: 0 });
      dirty.current = true;
    } else if (isPanning.current) {
      transform.current.tx = cx - panStart.current.x;
      transform.current.ty = cy - panStart.current.y;
      dirty.current = true;
    } else {
      const prev = hoveredId.current;
      const hit = hitTest(cx, cy);
      hoveredId.current = hit?.id ?? null;
      if (hoveredId.current !== prev) dirty.current = true;
    }
  }

  function handleMouseUp() {
    dragging.current = null;
    isPanning.current = false;
  }

  function handleMouseLeave() {
    dragging.current = null;
    isPanning.current = false;
    if (hoveredId.current) {
      hoveredId.current = null;
      dirty.current = true;
    }
  }

  function handleWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    const newScale = Math.max(0.2, Math.min(5, transform.current.scale * factor));

    const { tx, ty, scale } = transform.current;
    transform.current.tx = cx - (cx - tx) * (newScale / scale);
    transform.current.ty = cy - (cy - ty) * (newScale / scale);
    transform.current.scale = newScale;
    dirty.current = true;
  }

  function handleDoubleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = hitTest(cx, cy);
    if (!hit) return;

    const pos = posRef.current.get(hit.id);
    if (!pos) return;

    const targetScale = 2.0;
    const canvasW = canvasRef.current!.width;
    const canvasH = canvasRef.current!.height;
    const targetTx = canvasW / 2 - pos.x * targetScale;
    const targetTy = canvasH / 2 - pos.y * targetScale;

    const startTx = transform.current.tx;
    const startTy = transform.current.ty;
    const startScale = transform.current.scale;
    let frame = 0;
    const totalFrames = 20;

    function animateZoom() {
      frame++;
      const t = frame / totalFrames;
      const ease = t * (2 - t);
      transform.current.tx = startTx + (targetTx - startTx) * ease;
      transform.current.ty = startTy + (targetTy - startTy) * ease;
      transform.current.scale = startScale + (targetScale - startScale) * ease;
      dirty.current = true;
      if (frame < totalFrames) requestAnimationFrame(animateZoom);
    }
    requestAnimationFrame(animateZoom);
  }

  function handleZoomIn() {
    const W = canvasRef.current?.width ?? 800;
    const H = canvasRef.current?.height ?? 600;
    const factor = 1.3;
    const newScale = Math.min(5, transform.current.scale * factor);
    const { tx, ty, scale } = transform.current;
    transform.current.tx = W / 2 - (W / 2 - tx) * (newScale / scale);
    transform.current.ty = H / 2 - (H / 2 - ty) * (newScale / scale);
    transform.current.scale = newScale;
    dirty.current = true;
  }

  function handleZoomOut() {
    const W = canvasRef.current?.width ?? 800;
    const H = canvasRef.current?.height ?? 600;
    const factor = 0.7;
    const newScale = Math.max(0.2, transform.current.scale * factor);
    const { tx, ty, scale } = transform.current;
    transform.current.tx = W / 2 - (W / 2 - tx) * (newScale / scale);
    transform.current.ty = H / 2 - (H / 2 - ty) * (newScale / scale);
    transform.current.scale = newScale;
    dirty.current = true;
  }

  function handleZoomFit() {
    transform.current = { scale: 1, tx: 0, ty: 0 };
    dirty.current = true;
  }

  // Resize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      dirty.current = true;
    });
    observer.observe(canvas);
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    return () => observer.disconnect();
  }, []);

  const typeEntries = Object.entries(TYPE_COLORS).filter(([k]) => k !== "Default");

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center gap-3">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search nodes..."
          className="flex-1 px-3 py-1.5 rounded border text-sm outline-none"
          style={{
            background: "var(--card)",
            borderColor: "var(--border)",
            color: "var(--foreground)",
          }}
        />
        <div className="flex flex-wrap gap-2">
          {typeEntries.map(([type, color]) => (
            <span key={type} className="flex items-center gap-1 text-xs" style={{ color: "var(--muted)" }}>
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: color.fill }} />
              {type}
            </span>
          ))}
        </div>
      </div>

      <div className="flex gap-3 flex-1 min-h-[500px]">
        <div className="relative flex-1">
          <canvas
            ref={canvasRef}
            className="w-full h-full rounded-lg border cursor-grab active:cursor-grabbing"
            style={{ borderColor: "var(--border)", background: "#f8fafc" }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          />
          {/* Zoom controls */}
          <div
            className="absolute bottom-3 right-3 flex gap-1 rounded-lg border overflow-hidden"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
          >
            <button
              onClick={handleZoomIn}
              className="px-2.5 py-1.5 text-xs font-medium transition-colors hover:opacity-70"
              style={{ color: "var(--foreground)" }}
            >
              +
            </button>
            <button
              onClick={handleZoomOut}
              className="px-2.5 py-1.5 text-xs font-medium transition-colors hover:opacity-70 border-l border-r"
              style={{ color: "var(--foreground)", borderColor: "var(--border)" }}
            >
              &minus;
            </button>
            <button
              onClick={handleZoomFit}
              className="px-2.5 py-1.5 text-xs font-medium transition-colors hover:opacity-70"
              style={{ color: "var(--muted)" }}
            >
              Fit
            </button>
          </div>
        </div>

        {selected && (
          <div
            className="w-64 rounded-lg border p-4 overflow-y-auto shrink-0"
            style={{ borderColor: "var(--border)", background: "var(--card)" }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span
                className="w-3 h-3 rounded-full"
                style={{ background: getColor(selected.type).fill }}
              />
              <span className="font-semibold text-sm" style={{ color: "var(--foreground)" }}>
                {selected.label}
              </span>
            </div>
            <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>
              Type: {selected.type}
            </p>
            <p className="text-xs mb-3" style={{ color: "var(--muted)" }}>
              Connections: {degreeMap.get(selected.id) ?? 0}
            </p>
            {selected.properties && Object.keys(selected.properties).length > 0 && (
              <div>
                <p className="text-xs font-semibold mb-1" style={{ color: "var(--muted)" }}>
                  Properties
                </p>
                {Object.entries(selected.properties).map(([k, v]) => (
                  <div key={k} className="text-xs mb-1">
                    <span style={{ color: "var(--muted)" }}>{k}: </span>
                    <span className="mono" style={{ color: "var(--foreground)" }}>
                      {v}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <button
              onClick={() => setSelected(null)}
              className="mt-3 text-xs" style={{ color: "var(--muted)" }}
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
