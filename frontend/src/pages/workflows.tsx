// ============================================
// CognitionOS - Workflows Page
// ============================================

import { useState } from 'react';
import {
  GitBranch, Plus, Play, Pause, Settings, Clock, CheckCircle2,
  ArrowRight, Zap, RefreshCw, Trash2, Copy, Eye, Search,
  AlertCircle, MoreVertical, ArrowDown, ChevronDown,
} from 'lucide-react';
import { PageHeader, StatusBadge, Tabs, EmptyState, Modal, StatCard } from '../components/ui';
import { formatNumber, formatRelativeTime, formatPercent } from '../lib/utils';

const demoWorkflows = [
  {
    id: '1', name: 'Code Review Pipeline', description: 'Automated code review with linting, security scanning, and architecture validation',
    status: 'active', steps: [
      { id: 's1', name: 'Lint Code', type: 'agent', status: 'completed' },
      { id: 's2', name: 'Security Scan', type: 'agent', status: 'completed' },
      { id: 's3', name: 'Architecture Check', type: 'condition', status: 'running' },
      { id: 's4', name: 'Generate Report', type: 'transform', status: 'pending' },
      { id: 's5', name: 'Notify Team', type: 'notification', status: 'pending' },
    ],
    trigger: { type: 'webhook', config: { event: 'pull_request' } },
    last_run_at: '2026-03-01T06:30:00Z', run_count: 234, success_rate: 0.97,
    created_at: '2026-01-15T10:00:00Z',
  },
  {
    id: '2', name: 'Feature Build Pipeline', description: 'End-to-end feature implementation from requirements to deployed code',
    status: 'active', steps: [
      { id: 's1', name: 'Parse Requirements', type: 'agent', status: 'completed' },
      { id: 's2', name: 'Design Architecture', type: 'agent', status: 'completed' },
      { id: 's3', name: 'Generate Code', type: 'agent', status: 'completed' },
      { id: 's4', name: 'Run Tests', type: 'agent', status: 'completed' },
      { id: 's5', name: 'Deploy', type: 'api_call', status: 'completed' },
    ],
    trigger: { type: 'manual', config: {} },
    last_run_at: '2026-03-01T05:00:00Z', run_count: 89, success_rate: 0.92,
    created_at: '2026-02-01T10:00:00Z',
  },
  {
    id: '3', name: 'Nightly Performance Scan', description: 'Scheduled performance benchmarking with regression detection and alerting',
    status: 'active', steps: [
      { id: 's1', name: 'Run Benchmarks', type: 'agent', status: 'pending' },
      { id: 's2', name: 'Compare Results', type: 'transform', status: 'pending' },
      { id: 's3', name: 'Detect Regressions', type: 'condition', status: 'pending' },
      { id: 's4', name: 'Alert on Issues', type: 'notification', status: 'pending' },
    ],
    trigger: { type: 'schedule', config: { cron: '0 2 * * *' } },
    last_run_at: '2026-03-01T02:00:00Z', run_count: 45, success_rate: 0.98,
    created_at: '2026-02-15T10:00:00Z',
  },
  {
    id: '4', name: 'Documentation Generator', description: 'Auto-generate API docs, README, and changelogs from codebase analysis',
    status: 'paused', steps: [
      { id: 's1', name: 'Analyze Codebase', type: 'agent', status: 'pending' },
      { id: 's2', name: 'Generate Docs', type: 'agent', status: 'pending' },
      { id: 's3', name: 'Publish', type: 'api_call', status: 'pending' },
    ],
    trigger: { type: 'event', config: { event: 'release' } },
    last_run_at: '2026-02-28T12:00:00Z', run_count: 23, success_rate: 0.95,
    created_at: '2026-02-10T10:00:00Z',
  },
];

const stepTypeColors: Record<string, string> = {
  agent: 'bg-indigo-500/15 text-indigo-400 border-indigo-500/20',
  condition: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
  transform: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/20',
  api_call: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  notification: 'bg-violet-500/15 text-violet-400 border-violet-500/20',
};

const triggerLabels: Record<string, string> = {
  manual: 'Manual Trigger',
  schedule: 'Scheduled',
  webhook: 'Webhook',
  event: 'Event-Based',
};

export default function WorkflowsPage() {
  const [activeTab, setActiveTab] = useState('all');
  const [selectedWorkflow, setSelectedWorkflow] = useState<any>(null);

  const activeCount = demoWorkflows.filter(w => w.status === 'active').length;
  const totalRuns = demoWorkflows.reduce((s, w) => s + w.run_count, 0);
  const avgSuccess = demoWorkflows.reduce((s, w) => s + w.success_rate, 0) / demoWorkflows.length;

  const filteredWorkflows = demoWorkflows.filter(w => {
    if (activeTab === 'all') return true;
    return w.status === activeTab;
  });

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Workflows"
        description="Design and manage automated multi-agent pipelines"
        actions={
          <button className="btn btn-primary btn-md">
            <Plus className="w-4 h-4" />
            Create Workflow
          </button>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Workflows" value={demoWorkflows.length} icon={<GitBranch className="w-5 h-5" />} color="brand" />
        <StatCard title="Active" value={activeCount} icon={<Play className="w-5 h-5" />} color="emerald" />
        <StatCard title="Total Runs" value={formatNumber(totalRuns)} icon={<RefreshCw className="w-5 h-5" />} color="cyan" />
        <StatCard title="Avg Success" value={formatPercent(avgSuccess)} icon={<CheckCircle2 className="w-5 h-5" />} color="violet" />
      </div>

      <Tabs
        tabs={[
          { id: 'all', label: 'All', count: demoWorkflows.length },
          { id: 'active', label: 'Active', count: activeCount },
          { id: 'paused', label: 'Paused', count: demoWorkflows.filter(w => w.status === 'paused').length },
        ]}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Workflow List */}
      <div className="space-y-4">
        {filteredWorkflows.map((workflow) => (
          <div
            key={workflow.id}
            className="premium-card hover:border-brand-500/30 cursor-pointer transition-all"
            onClick={() => setSelectedWorkflow(workflow)}
          >
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div className="flex items-start gap-4 flex-1 min-w-0">
                <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-brand-600 to-violet-600 flex items-center justify-center flex-shrink-0">
                  <GitBranch className="w-5 h-5 text-white" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="font-semibold text-white truncate">{workflow.name}</h3>
                    <StatusBadge status={workflow.status} />
                  </div>
                  <p className="text-sm text-zinc-400 truncate">{workflow.description}</p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-zinc-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {triggerLabels[workflow.trigger.type]}
                    </span>
                    <span>{formatNumber(workflow.run_count)} runs</span>
                    <span className="text-emerald-400">{formatPercent(workflow.success_rate)} success</span>
                    {workflow.last_run_at && (
                      <span>Last run {formatRelativeTime(workflow.last_run_at)}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Steps Preview */}
              <div className="flex items-center gap-1 flex-shrink-0">
                {workflow.steps.map((step, idx) => (
                  <div key={step.id} className="flex items-center">
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium border ${stepTypeColors[step.type]}`}
                      title={`${step.name} (${step.type})`}
                    >
                      {idx + 1}
                    </div>
                    {idx < workflow.steps.length - 1 && (
                      <ArrowRight className="w-3 h-3 text-zinc-600 mx-0.5" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Workflow Detail Modal */}
      <Modal
        open={!!selectedWorkflow}
        onClose={() => setSelectedWorkflow(null)}
        title={selectedWorkflow?.name || ''}
        description={selectedWorkflow?.description}
        size="lg"
        footer={
          <>
            <button className="btn btn-secondary btn-md" onClick={() => setSelectedWorkflow(null)}>Close</button>
            <button className="btn btn-primary btn-md">
              <Play className="w-4 h-4" /> Run Now
            </button>
          </>
        }
      >
        {selectedWorkflow && (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Runs</p>
                <p className="text-lg font-bold text-white">{formatNumber(selectedWorkflow.run_count)}</p>
              </div>
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Success Rate</p>
                <p className="text-lg font-bold text-emerald-400">{formatPercent(selectedWorkflow.success_rate)}</p>
              </div>
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Trigger</p>
                <p className="text-sm font-medium text-white mt-1">{triggerLabels[selectedWorkflow.trigger.type]}</p>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">Pipeline Steps</h3>
              <div className="space-y-3">
                {selectedWorkflow.steps.map((step: any, idx: number) => (
                  <div key={step.id} className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm border ${stepTypeColors[step.type]}`}>
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-white">{step.name}</p>
                      <p className="text-xs text-zinc-500 capitalize">{step.type.replace('_', ' ')}</p>
                    </div>
                    <StatusBadge status={step.status} />
                    {idx < selectedWorkflow.steps.length - 1 && (
                      <div className="absolute left-5 mt-10 h-3 w-0.5 bg-zinc-700" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
