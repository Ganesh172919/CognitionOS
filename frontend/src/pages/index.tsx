/**
 * CognitionOS Main Dashboard
 *
 * Provides real-time visibility into:
 * - Agent reasoning and execution
 * - Task progress and timelines
 * - System health and metrics
 * - Memory usage and performance
 */

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Activity, AlertCircle, CheckCircle2, Clock, Cpu, Database, TrendingUp } from 'lucide-react';

import ReasoningVisualization from '../components/ReasoningVisualization';
import TaskGraph from '../components/TaskGraph';
import ExecutionTimeline from '../components/ExecutionTimeline';
import MemoryDashboard from '../components/MemoryDashboard';
import MetricsPanel from '../components/MetricsPanel';

const OBSERVABILITY_URL = process.env.OBSERVABILITY_URL || 'http://localhost:8009';
const EXPLAINABILITY_URL = process.env.EXPLAINABILITY_URL || 'http://localhost:8008';

export default function Dashboard() {
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'overview' | 'task-detail'>('overview');

  // Fetch dashboard data from observability service
  const { data: dashboardData, isLoading: loadingDashboard } = useQuery({
    queryKey: ['dashboard'],
    queryFn: async () => {
      const response = await axios.get(`${OBSERVABILITY_URL}/dashboard`);
      return response.data;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch active tasks
  const { data: activeTasks } = useQuery({
    queryKey: ['active-tasks'],
    queryFn: async () => {
      const response = await axios.get(`${process.env.API_GATEWAY_URL}/tasks/active`);
      return response.data;
    },
    refetchInterval: 3000,
  });

  // Fetch task explanation when task is selected
  const { data: taskExplanation } = useQuery({
    queryKey: ['task-explanation', selectedTaskId],
    queryFn: async () => {
      if (!selectedTaskId) return null;
      const response = await axios.post(`${EXPLAINABILITY_URL}/explain`, {
        task_id: selectedTaskId,
        level: 'detailed',
        include_timeline: true,
        include_reasoning: true,
        include_confidence: true,
      });
      return response.data;
    },
    enabled: !!selectedTaskId,
  });

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'text-green-600 bg-green-50';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50';
      case 'degraded':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy':
        return <CheckCircle2 className="w-5 h-5" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <Activity className="w-5 h-5" />;
    }
  };

  if (loadingDashboard) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">CognitionOS Dashboard</h1>
              <p className="text-sm text-gray-600 mt-1">Real-time agent monitoring and analysis</p>
            </div>

            {/* System Health Indicator */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${getHealthColor(dashboardData?.system_health)}`}>
              {getHealthIcon(dashboardData?.system_health)}
              <span className="font-medium capitalize">{dashboardData?.system_health || 'Unknown'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {viewMode === 'overview' ? (
          <div className="space-y-6">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="Active Tasks"
                value={activeTasks?.length || 0}
                icon={<Activity />}
                color="blue"
              />
              <MetricCard
                title="Total Tokens"
                value={dashboardData?.timeline_summary?.total_tokens?.toLocaleString() || '0'}
                icon={<Cpu />}
                color="purple"
              />
              <MetricCard
                title="Avg Error Rate"
                value={`${((dashboardData?.error_rates && Object.values(dashboardData.error_rates).reduce((a, b) => a + b, 0) / Object.keys(dashboardData.error_rates).length) * 100 || 0).toFixed(1)}%`}
                icon={<AlertCircle />}
                color="red"
              />
              <MetricCard
                title="Total Cost"
                value={`$${dashboardData?.timeline_summary?.total_cost_usd?.toFixed(4) || '0.0000'}`}
                icon={<TrendingUp />}
                color="green"
              />
            </div>

            {/* Active Alerts */}
            {dashboardData?.active_alerts && dashboardData.active_alerts.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  Active Alerts ({dashboardData.active_alerts.length})
                </h2>
                <div className="space-y-3">
                  {dashboardData.active_alerts.map((alert: any) => (
                    <div
                      key={alert.id}
                      className={`p-4 rounded-lg border ${
                        alert.severity === 'critical'
                          ? 'bg-red-50 border-red-200'
                          : alert.severity === 'warning'
                          ? 'bg-yellow-50 border-yellow-200'
                          : 'bg-blue-50 border-blue-200'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium text-gray-900">{alert.title}</h3>
                          <p className="text-sm text-gray-600 mt-1">{alert.service}</p>
                        </div>
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded ${
                            alert.severity === 'critical'
                              ? 'bg-red-100 text-red-800'
                              : alert.severity === 'warning'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-blue-100 text-blue-800'
                          }`}
                        >
                          {alert.severity}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        {new Date(alert.triggered_at).toLocaleString()}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Active Tasks */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Active Tasks</h2>
                <div className="space-y-3">
                  {activeTasks?.map((task: any) => (
                    <div
                      key={task.id}
                      className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 cursor-pointer transition-colors"
                      onClick={() => {
                        setSelectedTaskId(task.id);
                        setViewMode('task-detail');
                      }}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium text-gray-900">{task.title || 'Untitled Task'}</h3>
                          <p className="text-sm text-gray-600 mt-1">{task.description}</p>
                        </div>
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded ${
                            task.status === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : task.status === 'running'
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          {task.status}
                        </span>
                      </div>
                    </div>
                  ))}
                  {(!activeTasks || activeTasks.length === 0) && (
                    <p className="text-gray-500 text-center py-8">No active tasks</p>
                  )}
                </div>
              </div>

              {/* Recent Failures */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Failures</h2>
                <div className="space-y-3">
                  {dashboardData?.recent_failures?.slice(0, 5).map((failure: any, index: number) => (
                    <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium text-gray-900">{failure.service}</h3>
                          <p className="text-sm text-gray-600 mt-1">{failure.operation}</p>
                          {failure.error && (
                            <p className="text-xs text-red-600 mt-1">{failure.error}</p>
                          )}
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        {new Date(failure.timestamp).toLocaleString()}
                      </p>
                    </div>
                  ))}
                  {(!dashboardData?.recent_failures || dashboardData.recent_failures.length === 0) && (
                    <p className="text-gray-500 text-center py-8">No recent failures</p>
                  )}
                </div>
              </div>
            </div>

            {/* Metrics Panel */}
            <MetricsPanel dashboardData={dashboardData} />
          </div>
        ) : (
          /* Task Detail View */
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <button
                onClick={() => {
                  setViewMode('overview');
                  setSelectedTaskId(null);
                }}
                className="text-primary-600 hover:text-primary-700 font-medium"
              >
                ← Back to Overview
              </button>
            </div>

            {taskExplanation && (
              <>
                {/* Task Summary */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">Task Summary</h2>
                  <p className="text-gray-700">{taskExplanation.summary}</p>
                </div>

                {/* Reasoning Visualization */}
                {taskExplanation.reasoning_summary && (
                  <ReasoningVisualization reasoningData={taskExplanation.reasoning_summary} />
                )}

                {/* Execution Timeline */}
                {taskExplanation.timeline_summary && (
                  <ExecutionTimeline timelineData={taskExplanation.timeline_summary} />
                )}

                {/* Confidence Analysis */}
                {taskExplanation.confidence_analysis && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-4">Confidence Analysis</h2>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700">Overall Confidence</span>
                          <span className="text-lg font-bold text-gray-900">
                            {(taskExplanation.confidence_analysis.average_confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full transition-all"
                            style={{
                              width: `${taskExplanation.confidence_analysis.average_confidence * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="pt-4 border-t border-gray-200">
                        <p className="text-sm text-gray-700">
                          <strong>Recommendation:</strong> {taskExplanation.confidence_analysis.recommendation}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Metric Card Component
function MetricCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    purple: 'bg-purple-50 text-purple-600',
    red: 'bg-red-50 text-red-600',
    green: 'bg-green-50 text-green-600',
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-2">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}
