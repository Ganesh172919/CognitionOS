// ============================================
// CognitionOS - AI Agents Page
// ============================================

import { useState } from 'react';
import {
  Bot, Plus, Settings, Play, Pause, RefreshCw, Zap, Clock,
  CheckCircle2, XCircle, TrendingUp, Search, Filter, MoreVertical,
  Brain, Code2, FileSearch, Map, Network, Wrench, Eye,
} from 'lucide-react';
import { useAgents } from '../hooks/useApi';
import { StatCard, StatusBadge, PageHeader, Tabs, Modal, EmptyState, ProgressBar } from '../components/ui';
import { formatNumber, formatDuration, formatPercent } from '../lib/utils';

const AGENT_ICONS: Record<string, any> = {
  code_gen: Code2,
  research: FileSearch,
  planner: Map,
  reviewer: CheckCircle2,
  orchestrator: Network,
  custom: Wrench,
};

const AGENT_GRADIENTS: Record<string, string> = {
  code_gen: 'from-indigo-600 to-violet-600',
  research: 'from-cyan-600 to-blue-600',
  planner: 'from-amber-600 to-orange-600',
  reviewer: 'from-emerald-600 to-teal-600',
  orchestrator: 'from-violet-600 to-purple-600',
  custom: 'from-zinc-600 to-zinc-500',
};

// Demo agents
const demoAgents = [
  { id: '1', name: 'CodeMaster Pro', type: 'code_gen', status: 'active', model: 'gemini-2.0-flash', capabilities: ['TypeScript', 'Python', 'React', 'FastAPI'], total_tasks: 1247, success_rate: 0.967, avg_latency_ms: 2340, description: 'Advanced code generation with architecture awareness' },
  { id: '2', name: 'DeepResearch', type: 'research', status: 'processing', model: 'gemini-2.0-pro', capabilities: ['Web Search', 'Paper Analysis', 'Summarization'], total_tasks: 892, success_rate: 0.945, avg_latency_ms: 5670, description: 'Deep research and knowledge synthesis agent' },
  { id: '3', name: 'ArchPlanner', type: 'planner', status: 'idle', model: 'gemini-2.0-flash', capabilities: ['Task Decomposition', 'Dependency Analysis', 'Resource Estimation'], total_tasks: 456, success_rate: 0.989, avg_latency_ms: 1890, description: 'Intelligent task planning and decomposition' },
  { id: '4', name: 'CodeReviewer', type: 'reviewer', status: 'active', model: 'gemini-2.0-pro', capabilities: ['Code Review', 'Security Scan', 'Best Practices'], total_tasks: 2341, success_rate: 0.978, avg_latency_ms: 1234, description: 'Automated code review with security analysis' },
  { id: '5', name: 'Orchestrator-X', type: 'orchestrator', status: 'active', model: 'gemini-2.0-pro', capabilities: ['Multi-agent Coordination', 'Pipeline Management', 'Error Recovery'], total_tasks: 3456, success_rate: 0.995, avg_latency_ms: 890, description: 'Master orchestrator for complex multi-step workflows' },
  { id: '6', name: 'DataPipeline', type: 'custom', status: 'disabled', model: 'gemini-2.0-flash', capabilities: ['ETL', 'Data Validation', 'Schema Generation'], total_tasks: 234, success_rate: 0.912, avg_latency_ms: 3450, description: 'Custom data pipeline and transformation agent' },
];

export default function AgentsPage() {
  const [activeTab, setActiveTab] = useState('all');
  const [selectedAgent, setSelectedAgent] = useState<any>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const { data: apiAgents } = useAgents();

  const agents = apiAgents || demoAgents;

  const filteredAgents = agents.filter((agent: any) => {
    const matchesSearch = agent.name.toLowerCase().includes(searchQuery.toLowerCase());
    if (activeTab === 'all') return matchesSearch;
    if (activeTab === 'active') return matchesSearch && (agent.status === 'active' || agent.status === 'processing');
    if (activeTab === 'idle') return matchesSearch && agent.status === 'idle';
    if (activeTab === 'disabled') return matchesSearch && agent.status === 'disabled';
    return matchesSearch;
  });

  const activeCount = agents.filter((a: any) => a.status === 'active' || a.status === 'processing').length;
  const totalTasks = agents.reduce((sum: number, a: any) => sum + (a.total_tasks || 0), 0);
  const avgSuccess = agents.reduce((sum: number, a: any) => sum + (a.success_rate || 0), 0) / agents.length;

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="AI Agents"
        description="Manage and monitor your autonomous AI agent fleet"
        actions={
          <button className="btn btn-primary btn-md">
            <Plus className="w-4 h-4" />
            Deploy Agent
          </button>
        }
      />

      {/* Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Agents" value={agents.length} icon={<Bot className="w-5 h-5" />} color="brand" />
        <StatCard title="Active Now" value={activeCount} icon={<Zap className="w-5 h-5" />} color="emerald" />
        <StatCard title="Tasks Completed" value={formatNumber(totalTasks)} icon={<CheckCircle2 className="w-5 h-5" />} color="cyan" />
        <StatCard title="Avg Success Rate" value={formatPercent(avgSuccess)} icon={<TrendingUp className="w-5 h-5" />} color="violet" />
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <Tabs
          tabs={[
            { id: 'all', label: 'All', count: agents.length },
            { id: 'active', label: 'Active', count: activeCount },
            { id: 'idle', label: 'Idle' },
            { id: 'disabled', label: 'Disabled' },
          ]}
          activeTab={activeTab}
          onChange={setActiveTab}
        />
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-base pl-10"
          />
        </div>
      </div>

      {/* Agent Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredAgents.map((agent: any) => {
          const Icon = AGENT_ICONS[agent.type] || Bot;
          const gradient = AGENT_GRADIENTS[agent.type] || AGENT_GRADIENTS.custom;

          return (
            <div
              key={agent.id}
              className="glow-card cursor-pointer group"
              onClick={() => { setSelectedAgent(agent); setShowConfig(true); }}
            >
              <div className="p-6">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg`}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white group-hover:text-brand-300 transition-colors">{agent.name}</h3>
                      <p className="text-xs text-zinc-500 font-mono">{agent.model}</p>
                    </div>
                  </div>
                  <StatusBadge status={agent.status} pulse />
                </div>

                {/* Description */}
                <p className="text-sm text-zinc-400 mb-4 line-clamp-2">{agent.description || 'No description'}</p>

                {/* Capabilities */}
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {(agent.capabilities || []).slice(0, 3).map((cap: string) => (
                    <span key={cap} className="text-2xs px-2 py-0.5 rounded-md bg-surface-3 text-zinc-400 border border-zinc-800">
                      {cap}
                    </span>
                  ))}
                  {(agent.capabilities || []).length > 3 && (
                    <span className="text-2xs px-2 py-0.5 rounded-md bg-surface-3 text-zinc-500">
                      +{agent.capabilities.length - 3}
                    </span>
                  )}
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-3 pt-4 border-t border-zinc-800/50">
                  <div>
                    <p className="text-xs text-zinc-500">Tasks</p>
                    <p className="text-sm font-semibold text-white">{formatNumber(agent.total_tasks || 0)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Success</p>
                    <p className="text-sm font-semibold text-emerald-400">{formatPercent(agent.success_rate || 0)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Latency</p>
                    <p className="text-sm font-semibold text-zinc-300">{formatDuration(agent.avg_latency_ms || 0)}</p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Agent Detail Modal */}
      <Modal
        open={showConfig}
        onClose={() => setShowConfig(false)}
        title={selectedAgent?.name || 'Agent Details'}
        description={selectedAgent?.description}
        size="lg"
        footer={
          <>
            <button className="btn btn-secondary btn-md" onClick={() => setShowConfig(false)}>Close</button>
            <button className="btn btn-primary btn-md">
              <Settings className="w-4 h-4" /> Configure
            </button>
          </>
        }
      >
        {selectedAgent && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-zinc-400 mb-1">Type</p>
                <p className="text-white capitalize">{selectedAgent.type?.replace('_', ' ')}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-1">Model</p>
                <p className="text-white font-mono text-sm">{selectedAgent.model}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-1">Status</p>
                <StatusBadge status={selectedAgent.status} size="md" />
              </div>
              <div>
                <p className="text-sm text-zinc-400 mb-1">Total Tasks</p>
                <p className="text-white font-semibold">{formatNumber(selectedAgent.total_tasks)}</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-zinc-400 mb-2">Performance</p>
              <div className="space-y-3">
                <ProgressBar value={selectedAgent.success_rate * 100} label="Success Rate" showValue color="emerald" />
                <ProgressBar value={Math.min((selectedAgent.avg_latency_ms / 10000) * 100, 100)} label="Latency (lower is better)" showValue color="brand" />
              </div>
            </div>
            <div>
              <p className="text-sm text-zinc-400 mb-2">Capabilities</p>
              <div className="flex flex-wrap gap-2">
                {(selectedAgent.capabilities || []).map((cap: string) => (
                  <span key={cap} className="badge badge-primary">{cap}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
