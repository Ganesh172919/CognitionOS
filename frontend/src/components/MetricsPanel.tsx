/**
 * Metrics Panel Component
 *
 * Displays service-level metrics in a responsive grid.
 */

import { BarChart3, Activity, TrendingUp, AlertCircle } from 'lucide-react';
import { formatNumber } from '../lib/utils';
import { StatusBadge } from './ui';

interface ServiceMetrics {
  count: number;
  error_rate?: number;
  avg_latency_ms?: number;
  status?: string;
}

export default function MetricsPanel({ dashboardData }: { dashboardData: any }) {
  const services = dashboardData?.service_metrics || {};

  if (Object.keys(services).length === 0) {
    return (
      <div className="premium-card">
        <h2 className="section-header"><BarChart3 className="w-5 h-5 text-brand-400" /> Service Metrics</h2>
        <div className="text-center py-8 text-zinc-500 text-sm">No service metrics available</div>
      </div>
    );
  }

  return (
    <div className="premium-card">
      <h2 className="section-header"><BarChart3 className="w-5 h-5 text-brand-400" /> Service Metrics</h2>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {Object.entries(services).map(([service, metrics]: [string, any]) => (
          <div key={service} className="glass-card p-4 group">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-medium text-zinc-300 group-hover:text-white transition-colors truncate pr-2">
                {service}
              </h3>
              {metrics.status && <StatusBadge status={metrics.status} />}
            </div>
            <p className="text-2xl font-bold text-white">{formatNumber(metrics.count)}</p>
            <p className="text-xs text-zinc-500 mt-1">requests</p>
            {(metrics.avg_latency_ms || metrics.error_rate !== undefined) && (
              <div className="mt-3 pt-3 border-t border-zinc-800/50 flex items-center gap-3 text-xs">
                {metrics.avg_latency_ms && (
                  <span className="text-zinc-400">{metrics.avg_latency_ms}ms</span>
                )}
                {metrics.error_rate !== undefined && (
                  <span className={metrics.error_rate > 0.05 ? 'text-red-400' : 'text-emerald-400'}>
                    {(metrics.error_rate * 100).toFixed(1)}% err
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
