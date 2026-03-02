/**
 * Task Graph Component
 *
 * Visual task dependency graph with status indicators.
 */

import { useState } from 'react';
import { Network, ArrowRight, CheckCircle2, XCircle, Loader2, Clock, RefreshCw } from 'lucide-react';
import { StatusBadge } from './ui';

interface TaskNode {
  id: string;
  name: string;
  status: string;
  dependencies: string[];
  duration_ms?: number;
}

const demoTaskGraph: TaskNode[] = [
  { id: 't1', name: 'Parse Requirements', status: 'completed', dependencies: [], duration_ms: 1200 },
  { id: 't2', name: 'Design Architecture', status: 'completed', dependencies: ['t1'], duration_ms: 3400 },
  { id: 't3', name: 'Generate Models', status: 'completed', dependencies: ['t2'], duration_ms: 2100 },
  { id: 't4', name: 'Generate API Layer', status: 'running', dependencies: ['t2'], duration_ms: 4500 },
  { id: 't5', name: 'Generate Tests', status: 'pending', dependencies: ['t3', 't4'] },
  { id: 't6', name: 'Run Validation', status: 'pending', dependencies: ['t5'] },
  { id: 't7', name: 'Deploy', status: 'pending', dependencies: ['t6'] },
];

export default function TaskGraph() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'border-emerald-500 bg-emerald-500/10';
      case 'running': return 'border-brand-500 bg-brand-500/10 animate-glow';
      case 'failed': return 'border-red-500 bg-red-500/10';
      default: return 'border-zinc-700 bg-surface-3';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case 'running': return <Loader2 className="w-4 h-4 text-brand-400 animate-spin" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />;
      default: return <Clock className="w-4 h-4 text-zinc-500" />;
    }
  };

  // Group nodes by depth level
  const levels: TaskNode[][] = [];
  const placed = new Set<string>();
  const getLevel = (node: TaskNode): number => {
    if (node.dependencies.length === 0) return 0;
    return Math.max(...node.dependencies.map(dep => {
      const depNode = demoTaskGraph.find(n => n.id === dep);
      return depNode ? getLevel(depNode) + 1 : 0;
    }));
  };

  demoTaskGraph.forEach(node => {
    const level = getLevel(node);
    if (!levels[level]) levels[level] = [];
    levels[level].push(node);
  });

  return (
    <div className="premium-card">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
          <Network className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Task Dependency Graph</h2>
          <p className="text-xs text-zinc-400">Visual task flow and dependencies</p>
        </div>
      </div>

      {/* Graph Visualization */}
      <div className="flex items-start gap-4 overflow-x-auto pb-4">
        {levels.map((level, levelIdx) => (
          <div key={levelIdx} className="flex flex-col items-center gap-3 min-w-[140px]">
            <span className="text-2xs text-zinc-500 uppercase tracking-wider font-medium mb-1">
              Phase {levelIdx + 1}
            </span>
            {level.map((node) => (
              <div key={node.id} className="flex items-center">
                <div
                  className={`p-3 rounded-xl border-2 ${getStatusColor(node.status)} transition-all cursor-pointer hover:scale-105 min-w-[130px]`}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {getStatusIcon(node.status)}
                    <span className="text-xs font-semibold text-white">{node.name}</span>
                  </div>
                  {node.duration_ms && (
                    <span className="text-2xs text-zinc-500 font-mono">{node.duration_ms}ms</span>
                  )}
                </div>
                {levelIdx < levels.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-zinc-600 mx-2 flex-shrink-0" />
                )}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 pt-4 border-t border-zinc-800 mt-4">
        {[
          { status: 'Completed', color: 'bg-emerald-500' },
          { status: 'Running', color: 'bg-brand-500' },
          { status: 'Pending', color: 'bg-zinc-600' },
          { status: 'Failed', color: 'bg-red-500' },
        ].map(({ status, color }) => (
          <div key={status} className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span className={`w-2 h-2 rounded-full ${color}`} />
            {status}
          </div>
        ))}
      </div>
    </div>
  );
}
