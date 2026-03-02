// ============================================
// CognitionOS - Analytics Page
// ============================================

import { useState } from 'react';
import {
  BarChart3, TrendingUp, TrendingDown, Zap, DollarSign, Clock,
  Activity, Users, Globe, Cpu, ArrowUpRight, ArrowDownRight,
} from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Legend,
  CartesianGrid, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import { PageHeader, StatCard, Tabs, ProgressBar } from '../components/ui';
import { formatNumber, formatCurrency, formatPercent } from '../lib/utils';
import { CHART_COLORS, TIME_RANGES } from '../lib/constants';

// Generate demo data
const generateDailyData = () => Array.from({ length: 30 }, (_, i) => ({
  date: `Mar ${i + 1}`,
  tasks: Math.floor(Math.random() * 200 + 50),
  tokens: Math.floor(Math.random() * 500000 + 100000),
  cost: parseFloat((Math.random() * 50 + 10).toFixed(2)),
  apiCalls: Math.floor(Math.random() * 10000 + 2000),
  errors: Math.floor(Math.random() * 50 + 5),
  activeUsers: Math.floor(Math.random() * 500 + 100),
}));

const generateHourlyData = () => Array.from({ length: 24 }, (_, i) => ({
  hour: `${i.toString().padStart(2, '0')}:00`,
  latency: Math.floor(Math.random() * 300 + 80),
  throughput: Math.floor(Math.random() * 1000 + 200),
  p99: Math.floor(Math.random() * 800 + 200),
}));

const agentPerformance = [
  { agent: 'CodeMaster', tasks: 1247, success: 96.7, latency: 2340, tokens: 890000 },
  { agent: 'DeepResearch', tasks: 892, success: 94.5, latency: 5670, tokens: 1200000 },
  { agent: 'ArchPlanner', tasks: 456, success: 98.9, latency: 1890, tokens: 340000 },
  { agent: 'CodeReviewer', tasks: 2341, success: 97.8, latency: 1234, tokens: 567000 },
  { agent: 'Orchestrator-X', tasks: 3456, success: 99.5, latency: 890, tokens: 230000 },
];

const costBreakdown = [
  { name: 'Code Generation', value: 45, color: '#6366f1' },
  { name: 'Research', value: 25, color: '#06b6d4' },
  { name: 'Planning', value: 12, color: '#f59e0b' },
  { name: 'Review', value: 10, color: '#10b981' },
  { name: 'Orchestration', value: 8, color: '#8b5cf6' },
];

const radarData = [
  { metric: 'Speed', value: 85 },
  { metric: 'Accuracy', value: 92 },
  { metric: 'Coverage', value: 78 },
  { metric: 'Efficiency', value: 88 },
  { metric: 'Reliability', value: 95 },
  { metric: 'Scalability', value: 82 },
];

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('30d');
  const [activeTab, setActiveTab] = useState('overview');
  const dailyData = generateDailyData();
  const hourlyData = generateHourlyData();

  const totalTasks = dailyData.reduce((s, d) => s + d.tasks, 0);
  const totalTokens = dailyData.reduce((s, d) => s + d.tokens, 0);
  const totalCost = dailyData.reduce((s, d) => s + d.cost, 0);
  const totalApiCalls = dailyData.reduce((s, d) => s + d.apiCalls, 0);

  const tooltipStyle = {
    contentStyle: { background: '#1e1e23', border: '1px solid #27272a', borderRadius: '10px', fontSize: '12px' },
    labelStyle: { color: '#a1a1aa' },
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Analytics"
        description="Deep insights into platform performance, usage, and costs"
        actions={
          <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} className="input-base w-auto py-2 text-sm">
            {TIME_RANGES.map((r) => (
              <option key={r.value} value={r.value}>{r.label}</option>
            ))}
          </select>
        }
      />

      <Tabs
        tabs={[
          { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'performance', label: 'Performance', icon: <Activity className="w-4 h-4" /> },
          { id: 'costs', label: 'Cost Analysis', icon: <DollarSign className="w-4 h-4" /> },
          { id: 'agents', label: 'Agent Intel', icon: <Cpu className="w-4 h-4" /> },
        ]}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Top Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Total Tasks" value={formatNumber(totalTasks)} icon={<Activity className="w-5 h-5" />} color="brand" change={18} changePeriod="vs last period" />
            <StatCard title="Token Usage" value={formatNumber(totalTokens)} icon={<Zap className="w-5 h-5" />} color="amber" change={-12} changePeriod="vs last period" />
            <StatCard title="Total Cost" value={formatCurrency(totalCost)} icon={<DollarSign className="w-5 h-5" />} color="emerald" change={8} changePeriod="vs last period" />
            <StatCard title="API Calls" value={formatNumber(totalApiCalls)} icon={<Globe className="w-5 h-5" />} color="cyan" change={24} changePeriod="vs last period" />
          </div>

          {/* Task Volume Chart */}
          <div className="premium-card">
            <h2 className="section-header"><BarChart3 className="w-5 h-5 text-brand-400" /> Task Volume Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={dailyData}>
                <defs>
                  <linearGradient id="tasksGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" stroke="#52525b" fontSize={11} tickLine={false} />
                <YAxis stroke="#52525b" fontSize={11} tickLine={false} />
                <RechartsTooltip {...tooltipStyle} />
                <Area type="monotone" dataKey="tasks" stroke="#6366f1" fill="url(#tasksGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Token & Cost Dual Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="premium-card">
              <h2 className="section-header"><Zap className="w-5 h-5 text-amber-400" /> Token Consumption</h2>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={dailyData.slice(-14)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} tickLine={false} />
                  <RechartsTooltip {...tooltipStyle} />
                  <Bar dataKey="tokens" fill="#f59e0b" radius={[4, 4, 0, 0]} opacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="premium-card">
              <h2 className="section-header"><DollarSign className="w-5 h-5 text-emerald-400" /> Daily Cost ($)</h2>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={dailyData.slice(-14)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} tickLine={false} />
                  <RechartsTooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="cost" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981', r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'performance' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="premium-card">
              <h2 className="section-header"><Activity className="w-5 h-5 text-brand-400" /> Latency Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={hourlyData}>
                  <defs>
                    <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="p99Grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="hour" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} tickLine={false} />
                  <RechartsTooltip {...tooltipStyle} />
                  <Area type="monotone" dataKey="latency" stroke="#6366f1" fill="url(#latGrad)" strokeWidth={2} name="Avg Latency (ms)" />
                  <Area type="monotone" dataKey="p99" stroke="#ef4444" fill="url(#p99Grad)" strokeWidth={1.5} strokeDasharray="4 4" name="P99 Latency (ms)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="premium-card">
              <h2 className="section-header"><TrendingUp className="w-5 h-5 text-cyan-400" /> System Quality Radar</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#27272a" />
                  <PolarAngleAxis dataKey="metric" stroke="#71717a" fontSize={12} />
                  <PolarRadiusAxis stroke="#27272a" fontSize={10} />
                  <Radar name="Score" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'costs' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="premium-card">
              <h2 className="section-header"><DollarSign className="w-5 h-5 text-emerald-400" /> Cost Breakdown</h2>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={costBreakdown} cx="50%" cy="50%" innerRadius={70} outerRadius={110} dataKey="value" paddingAngle={3}>
                    {costBreakdown.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} stroke="transparent" />
                    ))}
                  </Pie>
                  <RechartsTooltip {...tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 mt-4">
                {costBreakdown.map((item) => (
                  <div key={item.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                      <span className="text-sm text-zinc-300">{item.name}</span>
                    </div>
                    <span className="text-sm font-medium text-white">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="lg:col-span-2 premium-card">
              <h2 className="section-header"><TrendingUp className="w-5 h-5 text-amber-400" /> Cost Trend</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={dailyData}>
                  <defs>
                    <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} tickLine={false} />
                  <RechartsTooltip {...tooltipStyle} />
                  <Area type="monotone" dataKey="cost" stroke="#10b981" fill="url(#costGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'agents' && (
        <div className="space-y-6">
          <div className="premium-card p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-left">
                  <th className="px-6 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Agent</th>
                  <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Tasks</th>
                  <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Success Rate</th>
                  <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Avg Latency</th>
                  <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Tokens Used</th>
                  <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Efficiency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {agentPerformance.map((agent) => (
                  <tr key={agent.agent} className="table-row">
                    <td className="px-6 py-4 text-white font-medium">{agent.agent}</td>
                    <td className="px-4 py-4 text-zinc-300">{formatNumber(agent.tasks)}</td>
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-2">
                        <ProgressBar value={agent.success} size="sm" color={agent.success >= 97 ? 'emerald' : agent.success >= 95 ? 'amber' : 'red'} />
                        <span className="text-xs text-zinc-300 w-12">{agent.success}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-4 text-zinc-300 font-mono text-xs">{agent.latency}ms</td>
                    <td className="px-4 py-4 text-zinc-300">{formatNumber(agent.tokens)}</td>
                    <td className="px-4 py-4">
                      <span className={`badge ${agent.success >= 97 ? 'badge-success' : 'badge-warning'}`}>
                        {agent.success >= 97 ? 'Excellent' : 'Good'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
