"use client";

import { useEffect, useRef, useState } from "react";

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

const TYPE_COLORS: Record<string, string> = {
  Module: "#3b82f6",
  Class: "#22c55e",
  Function: "#f59e0b",
  Method: "#8b5cf6",
  Variable: "#ef4444",
  Exception: "#ec4899",
  Constant: "#06b6d4",
  Default: "#64748b",
};

// Lightweight force-directed graph using canvas — no heavy dependency
// Uses simplified Fruchterman-Reingold layout
export default function GraphViewer({ nodes, edges, onNodeClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const posRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const velRef = useRef<Map<string, { vx: number; vy: number }>>(new Map());
  const animRef = useRef<number>(0);
  const dragging = useRef<{ id: string; ox: number; oy: number } | null>(null);
  const transform = useRef({ scale: 1, tx: 0, ty: 0 });

  // Filter by search
  const visibleIds = new Set(
    search
      ? nodes.filter((n) => n.label.toLowerCase().includes(search.toLowerCase())).map((n) => n.id)
      : nodes.map((n) => n.id)
  );

  // Initialize positions
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;

    for (const node of nodes) {
      if (!posRef.current.has(node.id)) {
        posRef.current.set(node.id, {
          x: W / 2 + (Math.random() - 0.5) * 400,
          y: H / 2 + (Math.random() - 0.5) * 400,
        });
        velRef.current.set(node.id, { vx: 0, vy: 0 });
      }
    }
  }, [nodes]);

  // Force simulation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let tick = 0;

    function step() {
      const W = canvas!.width;
      const H = canvas!.height;
      const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.8;
      const cooling = Math.max(0.05, 1 - tick * 0.002);

      // Repulsion
      for (const a of nodes) {
        const pa = posRef.current.get(a.id)!;
        const va = velRef.current.get(a.id)!;
        for (const b of nodes) {
          if (a.id === b.id) continue;
          const pb = posRef.current.get(b.id)!;
          const dx = pa.x - pb.x;
          const dy = pa.y - pb.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = (k * k) / dist;
          va.vx += (dx / dist) * force * 0.01;
          va.vy += (dy / dist) * force * 0.01;
        }
      }

      // Attraction along edges
      for (const edge of edges) {
        const ps = posRef.current.get(edge.source);
        const pt = posRef.current.get(edge.target);
        const vs = velRef.current.get(edge.source);
        const vt = velRef.current.get(edge.target);
        if (!ps || !pt || !vs || !vt) continue;
        const dx = ps.x - pt.x;
        const dy = ps.y - pt.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist * dist) / k;
        const fx = (dx / dist) * force * 0.005;
        const fy = (dy / dist) * force * 0.005;
        vs.vx -= fx;
        vs.vy -= fy;
        vt.vx += fx;
        vt.vy += fy;
      }

      // Apply velocities with damping
      for (const node of nodes) {
        if (dragging.current?.id === node.id) continue;
        const p = posRef.current.get(node.id)!;
        const v = velRef.current.get(node.id)!;
        v.vx *= 0.85;
        v.vy *= 0.85;
        p.x += v.vx * cooling;
        p.y += v.vy * cooling;
        // Clamp to canvas
        p.x = Math.max(20, Math.min(W - 20, p.x));
        p.y = Math.max(20, Math.min(H - 20, p.y));
      }

      draw(ctx!, W, H);
      tick++;
      animRef.current = requestAnimationFrame(step);
    }

    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [nodes, edges, search, selected]); // eslint-disable-line react-hooks/exhaustive-deps

  function draw(ctx: CanvasRenderingContext2D, W: number, H: number) {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(transform.current.tx, transform.current.ty);
    ctx.scale(transform.current.scale, transform.current.scale);

    // Edges
    for (const edge of edges) {
      const ps = posRef.current.get(edge.source);
      const pt = posRef.current.get(edge.target);
      if (!ps || !pt) continue;
      const dim = !visibleIds.has(edge.source) && !visibleIds.has(edge.target);
      ctx.beginPath();
      ctx.moveTo(ps.x, ps.y);
      ctx.lineTo(pt.x, pt.y);
      ctx.strokeStyle = dim ? "#1e293b" : "#334155";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Nodes
    for (const node of nodes) {
      const p = posRef.current.get(node.id);
      if (!p) continue;
      const color = TYPE_COLORS[node.type] ?? TYPE_COLORS.Default;
      const dim = search && !visibleIds.has(node.id);
      const isSelected = selected?.id === node.id;

      ctx.beginPath();
      ctx.arc(p.x, p.y, isSelected ? 10 : 7, 0, Math.PI * 2);
      ctx.fillStyle = dim ? "#1e293b" : color;
      ctx.fill();
      if (isSelected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      if (!dim) {
        ctx.fillStyle = "#cbd5e1";
        ctx.font = "10px sans-serif";
        ctx.fillText(node.label.slice(0, 20), p.x + 10, p.y + 4);
      }
    }

    ctx.restore();
  }

  function hitTest(cx: number, cy: number) {
    const { scale, tx, ty } = transform.current;
    const wx = (cx - tx) / scale;
    const wy = (cy - ty) / scale;
    for (const node of nodes) {
      const p = posRef.current.get(node.id);
      if (!p) continue;
      const d = Math.sqrt((wx - p.x) ** 2 + (wy - p.y) ** 2);
      if (d < 12) return node;
    }
    return null;
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = hitTest(cx, cy);
    if (hit) {
      dragging.current = { id: hit.id, ox: cx, oy: cy };
      setSelected(hit);
      onNodeClick?.(hit);
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!dragging.current) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { scale, tx, ty } = transform.current;
    const wx = (cx - tx) / scale;
    const wy = (cy - ty) / scale;
    posRef.current.set(dragging.current.id, { x: wx, y: wy });
  }

  function handleMouseUp() {
    dragging.current = null;
  }

  function handleWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    transform.current.scale = Math.max(0.2, Math.min(5, transform.current.scale * factor));
  }

  // Resize canvas on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
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
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
              {type}
            </span>
          ))}
        </div>
      </div>

      <div className="flex gap-3 flex-1 min-h-[500px]">
        <canvas
          ref={canvasRef}
          className="flex-1 rounded-lg border cursor-grab active:cursor-grabbing"
          style={{ borderColor: "var(--border)", background: "#0a1628" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        />

        {selected && (
          <div
            className="w-64 rounded-lg border p-4 overflow-y-auto shrink-0"
            style={{ borderColor: "var(--border)", background: "var(--card)" }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span
                className="w-3 h-3 rounded-full"
                style={{ background: TYPE_COLORS[selected.type] ?? TYPE_COLORS.Default }}
              />
              <span className="font-semibold text-sm" style={{ color: "var(--foreground)" }}>
                {selected.label}
              </span>
            </div>
            <p className="text-xs mb-3" style={{ color: "var(--muted)" }}>
              Type: {selected.type}
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
