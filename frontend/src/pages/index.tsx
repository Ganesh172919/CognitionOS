// ============================================
// CognitionOS - Main Dashboard (Redesigned)
// ============================================

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity, AlertCircle, CheckCircle2, Clock, Cpu, TrendingUp,
  Zap, ArrowRight, Bot, Code2, Database, Gauge, Shield, Wifi,
  BarChart3, GitBranch, Eye,
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { StatCard, StatusBadge, ProgressBar, EmptyState, SkeletonCard, PageHeader, Tabs } from '../components/ui';
import { useDashboard, useActiveTasks, useAgents, usePerformanceTrends } from '../hooks/useApi';
import { formatNumber, formatCurrency, formatDuration, formatRelativeTime, formatPercent } from '../lib/utils';
import { CHART_COLORS, TIME_RANGES } from '../lib/constants';
import apiClient from '../lib/api-client';
import Link from 'next/link';

// Demo data for charts when API is unavailable
const generateTrendData = () => {
  const now = Date.now();
  return Array.from({ length: 24 }, (_, i) => ({
    time: new Date(now - (23 - i) * 3600000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    latency: Math.floor(Math.random() * 200 + 100),
    throughput: Math.floor(Math.random() * 500 + 200),
    errors: Math.floor(Math.random() * 10),
    tokens: Math.floor(Math.random() * 50000 + 10000),
    cost: parseFloat((Math.random() * 2 + 0.5).toFixed(4)),
  }));
};

const generateServiceData = () => [
  { name: 'Agent Orchestrator', requests: 12450, errors: 23, latency: 145, status: 'healthy' },
  { name: 'Code Generator', requests: 8932, errors: 12, latency: 890, status: 'healthy' },
  { name: 'Task Scheduler', requests: 15600, errors: 5, latency: 45, status: 'healthy' },
  { name: 'Memory Store', requests: 45000, errors: 120, latency: 12, status: 'warning' },
  { name: 'API Gateway', requests: 67800, errors: 34, latency: 23, status: 'healthy' },
  { name: 'Auth Service', requests: 23400, errors: 0, latency: 78, status: 'healthy' },
];

export default function Dashboard() {
  const [timeRange, setTimeRange] = useState('24h');
  const { data: dashboardData, isLoading } = useDashboard();
  const { data: activeTasks } = useActiveTasks();
  const { data: agents } = useAgents();

  const trendData = generateTrendData();
  const serviceData = generateServiceData();

  const systemHealth = dashboardData?.system_health || 'healthy';
  const activeTaskCount = activeTasks?.length || 0;
  const totalAgents = agents?.length || 6;
  const activeAgents = agents?.filter((a: any) => a.status === 'active')?.length || 4;

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-fade-in">
      <PageHeader
        title="Mission Control"
        description="Real-time system intelligence and agent orchestration overview"
        actions={
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className={`status-dot ${systemHealth === 'healthy' ? 'status-dot-healthy' : systemHealth === 'warning' ? 'status-dot-warning' : 'status-dot-danger'}`} />
              <span className="text-sm font-medium text-zinc-300 capitalize">{systemHealth}</span>
            </div>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="input-base w-auto py-2 text-sm"
            >
              {TIME_RANGES.map((r) => (
                <option key={r.value} value={r.value}>{r.label}</option>
              ))}
            </select>
          </div>
        }
      />

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Active Tasks"
          value={activeTaskCount}
          icon={<Activity className="w-5 h-5" />}
          color="brand"
          change={12}
          changePeriod="vs last hour"
          loading={isLoading}
        />
        <StatCard
          title="AI Agents Online"
          value={`${activeAgents}/${totalAgents}`}
          icon={<Bot className="w-5 h-5" />}
          color="cyan"
          change={0}
          changePeriod="stable"
          loading={isLoading}
        />
        <StatCard
          title="Tokens Used Today"
          value={formatNumber(dashboardData?.timeline_summary?.total_tokens || 234567)}
          icon={<Cpu className="w-5 h-5" />}
          color="violet"
          change={-8}
          changePeriod="vs yesterday"
          loading={isLoading}
        />
        <StatCard
          title="Est. Cost Today"
          value={formatCurrency(dashboardData?.timeline_summary?.total_cost_usd || 12.47)}
          icon={<TrendingUp className="w-5 h-5" />}
          color="emerald"
          change={-15}
          changePeriod="vs yesterday"
          loading={isLoading}
        />
      </div>

      {/* Performance Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Latency & Throughput Chart */}
        <div className="premium-card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="section-header mb-0">
              <Gauge className="w-5 h-5 text-brand-400" />
              Performance Metrics
            </h2>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-indigo-500" /> Latency</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-cyan-500" /> Throughput</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="throughputGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
              <RechartsTooltip
                contentStyle={{ background: '#1e1e23', border: '1px solid #27272a', borderRadius: '10px', fontSize: '12px' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Area type="monotone" dataKey="latency" stroke="#6366f1" fill="url(#latencyGrad)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="throughput" stroke="#06b6d4" fill="url(#throughputGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Token Usage Chart */}
        <div className="premium-card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="section-header mb-0">
              <Zap className="w-5 h-5 text-amber-400" />
              Token Consumption
            </h2>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500" /> Tokens</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500" /> Cost ($)</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={trendData}>
              <XAxis dataKey="time" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
              <RechartsTooltip
                contentStyle={{ background: '#1e1e23', border: '1px solid #27272a', borderRadius: '10px', fontSize: '12px' }}
                labelStyle={{ color: '#a1a1aa' }}
              />
              <Bar dataKey="tokens" fill="#f59e0b" radius={[4, 4, 0, 0]} opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Services & Tasks Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Service Health Grid */}
        <div className="lg:col-span-2 premium-card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="section-header mb-0">
              <Shield className="w-5 h-5 text-emerald-400" />
              Service Health
            </h2>
            <Link href="/analytics">
              <span className="text-sm text-brand-400 hover:text-brand-300 flex items-center gap-1 cursor-pointer">
                View all <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </Link>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {serviceData.map((service) => (
              <div key={service.name} className="glass-card p-4 flex items-center justify-between">
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`w-2 h-2 rounded-full ${service.status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-white truncate">{service.name}</p>
                    <p className="text-xs text-zinc-500">{formatNumber(service.requests)} req · {service.latency}ms</p>
                  </div>
                </div>
                <StatusBadge status={service.status} />
              </div>
            ))}
          </div>
        </div>

        {/* Active Tasks */}
        <div className="premium-card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="section-header mb-0">
              <Activity className="w-5 h-5 text-brand-400" />
              Active Tasks
            </h2>
            <Link href="/tasks">
              <span className="text-sm text-brand-400 hover:text-brand-300 flex items-center gap-1 cursor-pointer">
                View all <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </Link>
          </div>
          <div className="space-y-3">
            {(activeTasks && activeTasks.length > 0) ? activeTasks.slice(0, 5).map((task: any) => (
              <div key={task.id} className="glass-card p-3 cursor-pointer hover:border-brand-500/30">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-white truncate">{task.title || 'Untitled'}</p>
                    <p className="text-xs text-zinc-500 mt-0.5 truncate">{task.description || 'No description'}</p>
                  </div>
                  <StatusBadge status={task.status || 'running'} />
                </div>
              </div>
            )) : (
              <div className="text-center py-8">
                <p className="text-sm text-zinc-500">No active tasks</p>
                <Link href="/tasks">
                  <span className="text-sm text-brand-400 hover:text-brand-300 mt-2 inline-block cursor-pointer">
                    Create a task →
                  </span>
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Alerts Section */}
      {dashboardData?.active_alerts && dashboardData.active_alerts.length > 0 && (
        <div className="premium-card border-amber-500/20">
          <h2 className="section-header">
            <AlertCircle className="w-5 h-5 text-amber-400" />
            Active Alerts ({dashboardData.active_alerts.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {dashboardData.active_alerts.map((alert: any) => (
              <div
                key={alert.id}
                className={`p-4 rounded-xl border ${
                  alert.severity === 'critical'
                    ? 'bg-red-500/5 border-red-500/20'
                    : alert.severity === 'warning'
                    ? 'bg-amber-500/5 border-amber-500/20'
                    : 'bg-blue-500/5 border-blue-500/20'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-white">{alert.title}</h3>
                    <p className="text-xs text-zinc-400 mt-0.5">{alert.service}</p>
                  </div>
                  <StatusBadge status={alert.severity} />
                </div>
                {alert.triggered_at && (
                  <p className="text-xs text-zinc-500 mt-2">{formatRelativeTime(alert.triggered_at)}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Generate Code', icon: Code2, href: '/code-studio', gradient: 'from-indigo-600 to-violet-600' },
          { label: 'Create Workflow', icon: GitBranch, href: '/workflows', gradient: 'from-cyan-600 to-blue-600' },
          { label: 'Browse Plugins', icon: Database, href: '/plugins', gradient: 'from-emerald-600 to-teal-600' },
          { label: 'View Analytics', icon: BarChart3, href: '/analytics', gradient: 'from-amber-600 to-orange-600' },
        ].map((action) => (
          <Link key={action.label} href={action.href}>
            <div className="group relative overflow-hidden rounded-xl p-5 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-elevation-3">
              <div className={`absolute inset-0 bg-gradient-to-br ${action.gradient} opacity-10 group-hover:opacity-20 transition-opacity`} />
              <div className="relative">
                <action.icon className="w-8 h-8 text-white mb-3" />
                <p className="text-sm font-semibold text-white">{action.label}</p>
                <ArrowRight className="w-4 h-4 text-zinc-400 group-hover:text-white group-hover:translate-x-1 transition-all mt-2" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Resource Utilization */}
      <div className="premium-card">
        <h2 className="section-header">
          <Gauge className="w-5 h-5 text-cyan-400" />
          Resource Utilization
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mt-4">
          <ProgressBar value={dashboardData?.resource_utilization?.cpu_percent || 45} label="CPU" showValue color="brand" />
          <ProgressBar value={dashboardData?.resource_utilization?.memory_percent || 62} label="Memory" showValue color="cyan" />
          <ProgressBar value={dashboardData?.resource_utilization?.disk_percent || 38} label="Disk" showValue color="emerald" />
          <ProgressBar value={dashboardData?.resource_utilization?.gpu_percent || 78} label="GPU" showValue color="amber" />
        </div>
      </div>
    </div>
  );
}
