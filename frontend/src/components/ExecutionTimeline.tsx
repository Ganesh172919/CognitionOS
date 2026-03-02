/**
 * Execution Timeline Component
 *
 * Displays task execution timeline with animated step visualization.
 */

import { Clock, CheckCircle2, XCircle, Loader2, AlertCircle } from 'lucide-react';
import { formatDuration } from '../lib/utils';

interface TimelineEvent {
  name: string;
  type?: string;
  duration_ms: number;
  timestamp?: string;
  status?: string;
}

interface TimelineData {
  events?: TimelineEvent[];
  total_duration_ms?: number;
}

export default function ExecutionTimeline({ timelineData }: { timelineData: TimelineData }) {
  const events = timelineData?.events || [];
  const totalDuration = timelineData?.total_duration_ms || events.reduce((s, e) => s + e.duration_ms, 0);

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'running': return <Loader2 className="w-4 h-4 text-brand-400 animate-spin" />;
      default: return <Clock className="w-4 h-4 text-zinc-400" />;
    }
  };

  const getBarColor = (event: TimelineEvent) => {
    if (event.status === 'failed') return 'bg-red-500';
    if (event.status === 'running') return 'bg-brand-500 animate-pulse';
    const ratio = event.duration_ms / (totalDuration || 1);
    if (ratio > 0.5) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  return (
    <div className="premium-card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-600 to-blue-600 flex items-center justify-center">
            <Clock className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Execution Timeline</h2>
            <p className="text-xs text-zinc-400">Step-by-step execution trace</p>
          </div>
        </div>
        {totalDuration > 0 && (
          <span className="text-sm font-mono text-zinc-300">
            Total: {formatDuration(totalDuration)}
          </span>
        )}
      </div>

      {events.length > 0 ? (
        <div className="space-y-2">
          {events.map((event, i) => {
            const widthPct = totalDuration > 0 ? Math.max((event.duration_ms / totalDuration) * 100, 5) : 50;
            return (
              <div
                key={i}
                className="flex items-center gap-4 p-3 rounded-xl hover:bg-surface-3/50 transition-colors group"
                style={{ animationDelay: `${i * 0.05}s` }}
              >
                {/* Step number */}
                <span className="w-6 h-6 rounded-md bg-surface-3 flex items-center justify-center text-2xs font-bold text-zinc-400 flex-shrink-0">
                  {i + 1}
                </span>

                {/* Status icon */}
                <div className="flex-shrink-0">{getStatusIcon(event.status)}</div>

                {/* Name */}
                <span className="text-sm text-zinc-300 w-40 truncate flex-shrink-0 group-hover:text-white transition-colors">
                  {event.name}
                </span>

                {/* Duration bar */}
                <div className="flex-1 bg-surface-3 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-700 ease-out ${getBarColor(event)}`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>

                {/* Duration text */}
                <span className="text-xs font-mono text-zinc-500 w-16 text-right flex-shrink-0">
                  {formatDuration(event.duration_ms)}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8 text-zinc-500 text-sm">
          No timeline data available
        </div>
      )}
    </div>
  );
}
