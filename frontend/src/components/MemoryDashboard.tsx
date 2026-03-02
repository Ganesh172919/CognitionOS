/**
 * Memory Dashboard Component
 *
 * Displays AI agent memory usage and context management.
 */

import { Database, HardDrive, RefreshCw, Trash2, Plus, Search } from 'lucide-react';
import { ProgressBar } from './ui';
import { formatBytes, formatNumber } from '../lib/utils';

const demoMemoryData = {
  total_entries: 15234,
  total_size_bytes: 524288000,
  active_contexts: 12,
  hit_rate: 0.94,
  categories: [
    { name: 'Code Context', entries: 5678, size_bytes: 180000000, hit_rate: 0.96 },
    { name: 'Task History', entries: 4523, size_bytes: 150000000, hit_rate: 0.92 },
    { name: 'User Preferences', entries: 234, size_bytes: 5000000, hit_rate: 0.99 },
    { name: 'Agent State', entries: 3456, size_bytes: 120000000, hit_rate: 0.88 },
    { name: 'Embeddings Cache', entries: 1343, size_bytes: 69288000, hit_rate: 0.95 },
  ],
};

export default function MemoryDashboard() {
  return (
    <div className="premium-card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-600 to-teal-600 flex items-center justify-center">
            <Database className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Memory Dashboard</h2>
            <p className="text-xs text-zinc-400">Agent memory and context management</p>
          </div>
        </div>
        <button className="btn btn-ghost btn-sm text-zinc-400">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        <div className="glass-card p-3 text-center">
          <p className="text-xs text-zinc-400">Total Entries</p>
          <p className="text-lg font-bold text-white">{formatNumber(demoMemoryData.total_entries)}</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xs text-zinc-400">Total Size</p>
          <p className="text-lg font-bold text-white">{formatBytes(demoMemoryData.total_size_bytes)}</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xs text-zinc-400">Active Contexts</p>
          <p className="text-lg font-bold text-cyan-400">{demoMemoryData.active_contexts}</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xs text-zinc-400">Cache Hit Rate</p>
          <p className="text-lg font-bold text-emerald-400">{(demoMemoryData.hit_rate * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="space-y-3">
        {demoMemoryData.categories.map((cat) => {
          const sizePct = (cat.size_bytes / demoMemoryData.total_size_bytes) * 100;
          return (
            <div key={cat.name} className="glass-card p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <HardDrive className="w-4 h-4 text-zinc-400" />
                  <span className="text-sm font-medium text-white">{cat.name}</span>
                </div>
                <div className="flex items-center gap-4 text-xs text-zinc-400">
                  <span>{formatNumber(cat.entries)} entries</span>
                  <span>{formatBytes(cat.size_bytes)}</span>
                  <span className="text-emerald-400">{(cat.hit_rate * 100).toFixed(0)}% hit</span>
                </div>
              </div>
              <div className="w-full bg-surface-3 rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-1.5 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-700"
                  style={{ width: `${sizePct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
