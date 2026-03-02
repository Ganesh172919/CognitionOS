// ============================================
// CognitionOS - Tasks Management Page
// ============================================

import { useState } from 'react';
import {
  ListTodo, Plus, Search, Filter, Clock, CheckCircle2, XCircle,
  Loader2, AlertCircle, Ban, RefreshCw, ArrowUpRight, Eye,
  MoreVertical, Trash2, Play, Pause,
} from 'lucide-react';
import { useAllTasks, useActiveTasks, useCreateTask } from '../hooks/useApi';
import { PageHeader, StatusBadge, Tabs, EmptyState, Modal, StatCard } from '../components/ui';
import { formatRelativeTime, formatDuration, formatNumber, formatCurrency } from '../lib/utils';

const demoTasks = [
  { id: '1', title: 'Generate User Authentication Module', description: 'Create JWT-based auth with RBAC and refresh token rotation', status: 'completed', priority: 'high', agent_id: '1', created_at: '2026-03-01T01:00:00Z', completed_at: '2026-03-01T01:05:23Z', duration_ms: 323000, tokens_used: 45000, cost_usd: 0.0234 },
  { id: '2', title: 'Build Real-time Notification System', description: 'WebSocket-based notification system with event queuing and delivery tracking', status: 'running', priority: 'high', agent_id: '2', created_at: '2026-03-01T01:30:00Z', started_at: '2026-03-01T01:30:15Z', tokens_used: 12000, cost_usd: 0.0067 },
  { id: '3', title: 'Database Schema Migration', description: 'Migrate user tables to support multi-tenancy with row-level security', status: 'pending', priority: 'critical', created_at: '2026-03-01T01:45:00Z' },
  { id: '4', title: 'API Rate Limiter Implementation', description: 'Token bucket rate limiting with Redis backend and per-user quotas', status: 'queued', priority: 'medium', agent_id: '3', created_at: '2026-03-01T01:50:00Z' },
  { id: '5', title: 'Frontend Dashboard Redesign', description: 'Dark theme SaaS dashboard with real-time charts and responsive layout', status: 'completed', priority: 'medium', agent_id: '1', created_at: '2026-02-28T23:00:00Z', completed_at: '2026-03-01T00:12:00Z', duration_ms: 4320000, tokens_used: 89000, cost_usd: 0.0456 },
  { id: '6', title: 'Payment Integration with Stripe', description: 'Complete Stripe integration with subscription management and webhook handling', status: 'failed', priority: 'high', agent_id: '2', created_at: '2026-02-28T22:00:00Z', duration_ms: 120000, tokens_used: 23000, cost_usd: 0.012 },
  { id: '7', title: 'Automated Testing Pipeline', description: 'Set up Jest, Playwright, and k6 for unit, e2e, and performance testing', status: 'retrying', priority: 'low', agent_id: '4', created_at: '2026-02-28T21:00:00Z', tokens_used: 34000, cost_usd: 0.018 },
  { id: '8', title: 'Documentation Generator', description: 'Auto-generate API docs from TypeScript types and decorators', status: 'completed', priority: 'low', agent_id: '1', created_at: '2026-02-28T20:00:00Z', completed_at: '2026-02-28T20:45:00Z', duration_ms: 2700000, tokens_used: 56000, cost_usd: 0.029 },
];

const priorityColors: Record<string, string> = {
  critical: 'text-red-400 bg-red-500/10 border-red-500/20',
  high: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  medium: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  low: 'text-zinc-400 bg-zinc-500/10 border-zinc-500/20',
};

export default function TasksPage() {
  const [activeTab, setActiveTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newPriority, setNewPriority] = useState('medium');
  const { data: apiTasks } = useAllTasks();
  const createMutation = useCreateTask();

  const tasks = apiTasks || demoTasks;

  const filteredTasks = tasks.filter((task: any) => {
    const matchesSearch = task.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.description?.toLowerCase().includes(searchQuery.toLowerCase());
    if (activeTab === 'all') return matchesSearch;
    return matchesSearch && task.status === activeTab;
  });

  const statusCounts = {
    running: tasks.filter((t: any) => t.status === 'running').length,
    completed: tasks.filter((t: any) => t.status === 'completed').length,
    failed: tasks.filter((t: any) => t.status === 'failed').length,
    pending: tasks.filter((t: any) => t.status === 'pending' || t.status === 'queued').length,
  };

  const totalTokens = tasks.reduce((sum: number, t: any) => sum + (t.tokens_used || 0), 0);
  const totalCost = tasks.reduce((sum: number, t: any) => sum + (t.cost_usd || 0), 0);

  const handleCreate = async () => {
    if (!newTitle.trim()) return;
    try {
      await createMutation.mutateAsync({ title: newTitle, description: newDesc, priority: newPriority });
    } catch (e) {
      // Demo mode
    }
    setShowCreate(false);
    setNewTitle('');
    setNewDesc('');
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Tasks"
        description="Monitor and manage all AI agent task executions"
        actions={
          <button onClick={() => setShowCreate(true)} className="btn btn-primary btn-md">
            <Plus className="w-4 h-4" />
            New Task
          </button>
        }
      />

      {/* Quick Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard title="Running" value={statusCounts.running} icon={<Loader2 className="w-5 h-5 animate-spin" />} color="brand" />
        <StatCard title="Completed" value={statusCounts.completed} icon={<CheckCircle2 className="w-5 h-5" />} color="emerald" />
        <StatCard title="Failed" value={statusCounts.failed} icon={<XCircle className="w-5 h-5" />} color="red" />
        <StatCard title="Tokens Used" value={formatNumber(totalTokens)} icon={<AlertCircle className="w-5 h-5" />} color="amber" />
        <StatCard title="Total Cost" value={formatCurrency(totalCost)} icon={<ArrowUpRight className="w-5 h-5" />} color="cyan" />
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <Tabs
          tabs={[
            { id: 'all', label: 'All', count: tasks.length },
            { id: 'running', label: 'Running', count: statusCounts.running },
            { id: 'completed', label: 'Completed', count: statusCounts.completed },
            { id: 'failed', label: 'Failed', count: statusCounts.failed },
          ]}
          activeTab={activeTab}
          onChange={setActiveTab}
        />
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            placeholder="Search tasks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-base pl-10"
          />
        </div>
      </div>

      {/* Task List */}
      <div className="premium-card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left">
                <th className="px-6 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Task</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Status</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Priority</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Tokens</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Cost</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Duration</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {filteredTasks.map((task: any) => (
                <tr key={task.id} className="table-row group cursor-pointer">
                  <td className="px-6 py-4">
                    <div className="min-w-0">
                      <p className="text-white font-medium truncate max-w-xs group-hover:text-brand-300 transition-colors">{task.title}</p>
                      <p className="text-xs text-zinc-500 truncate max-w-xs mt-0.5">{task.description}</p>
                    </div>
                  </td>
                  <td className="px-4 py-4"><StatusBadge status={task.status} /></td>
                  <td className="px-4 py-4">
                    <span className={`badge border ${priorityColors[task.priority] || priorityColors.medium}`}>
                      {task.priority}
                    </span>
                  </td>
                  <td className="px-4 py-4">
                    <span className="text-zinc-300 font-mono text-xs">{task.tokens_used ? formatNumber(task.tokens_used) : '—'}</span>
                  </td>
                  <td className="px-4 py-4">
                    <span className="text-zinc-300 font-mono text-xs">{task.cost_usd ? formatCurrency(task.cost_usd) : '—'}</span>
                  </td>
                  <td className="px-4 py-4">
                    <span className="text-zinc-300 text-xs">{task.duration_ms ? formatDuration(task.duration_ms) : '—'}</span>
                  </td>
                  <td className="px-4 py-4">
                    <span className="text-zinc-400 text-xs">{formatRelativeTime(task.created_at)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filteredTasks.length === 0 && (
          <EmptyState
            icon={<ListTodo className="w-8 h-8" />}
            title="No tasks found"
            description="No tasks match your current filters. Try adjusting your search or create a new task."
            action={
              <button onClick={() => setShowCreate(true)} className="btn btn-primary btn-md">
                <Plus className="w-4 h-4" /> Create Task
              </button>
            }
          />
        )}
      </div>

      {/* Create Task Modal */}
      <Modal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="Create New Task"
        description="Define a task for an AI agent to execute"
        footer={
          <>
            <button className="btn btn-secondary btn-md" onClick={() => setShowCreate(false)}>Cancel</button>
            <button className="btn btn-primary btn-md" onClick={handleCreate} disabled={!newTitle.trim()}>
              <Plus className="w-4 h-4" /> Create Task
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Title</label>
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              placeholder="e.g., Build payment processing module"
              className="input-base"
            />
          </div>
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Description</label>
            <textarea
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="Describe the task requirements in detail..."
              className="input-base min-h-[100px] resize-none"
            />
          </div>
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Priority</label>
            <div className="flex gap-2">
              {['low', 'medium', 'high', 'critical'].map((p) => (
                <button
                  key={p}
                  onClick={() => setNewPriority(p)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all capitalize ${
                    newPriority === p
                      ? 'bg-brand-600 text-white'
                      : 'bg-surface-3 text-zinc-400 border border-zinc-800'
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Modal>
    </div>
  );
}
