// ============================================
// CognitionOS - Shared UI Components
// ============================================

import { ReactNode } from 'react';
import {
  CheckCircle2, AlertCircle, XCircle, Clock, Loader2, Ban,
  RefreshCw, TrendingUp, TrendingDown, Minus, ArrowUpRight,
  ArrowDownRight, Activity, Info,
} from 'lucide-react';

// ---- Stat Card ----

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: ReactNode;
  change?: number;
  changePeriod?: string;
  color?: 'brand' | 'cyan' | 'emerald' | 'amber' | 'red' | 'violet';
  loading?: boolean;
}

const colorMap = {
  brand: { bg: 'bg-indigo-500/10', text: 'text-indigo-400', border: 'border-indigo-500/20', glow: 'shadow-glow-brand' },
  cyan: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/20', glow: 'shadow-glow-accent' },
  emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/20', glow: 'shadow-glow-success' },
  amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/20', glow: '' },
  red: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/20', glow: 'shadow-glow-danger' },
  violet: { bg: 'bg-violet-500/10', text: 'text-violet-400', border: 'border-violet-500/20', glow: '' },
};

export function StatCard({ title, value, icon, change, changePeriod, color = 'brand', loading }: StatCardProps) {
  const c = colorMap[color];
  return (
    <div className={`premium-card group hover:${c.glow}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-zinc-400">{title}</p>
          {loading ? (
            <div className="skeleton h-8 w-24 mt-2" />
          ) : (
            <p className="text-2xl font-bold text-white mt-1 tracking-tight">{value}</p>
          )}
          {change !== undefined && (
            <div className="flex items-center gap-1 mt-2">
              {change > 0 ? (
                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400" />
              ) : change < 0 ? (
                <ArrowDownRight className="w-3.5 h-3.5 text-red-400" />
              ) : (
                <Minus className="w-3.5 h-3.5 text-zinc-500" />
              )}
              <span className={`text-xs font-medium ${change > 0 ? 'text-emerald-400' : change < 0 ? 'text-red-400' : 'text-zinc-500'}`}>
                {Math.abs(change)}%
              </span>
              {changePeriod && <span className="text-xs text-zinc-500">{changePeriod}</span>}
            </div>
          )}
        </div>
        {icon && (
          <div className={`p-3 rounded-xl ${c.bg} ${c.text} ${c.border} border`}>
            {icon}
          </div>
        )}
      </div>
      {/* Bottom gradient line */}
      <div className={`absolute bottom-0 left-0 right-0 h-0.5 ${c.bg} opacity-0 group-hover:opacity-100 transition-opacity`} />
    </div>
  );
}

// ---- Status Badge ----

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
  pulse?: boolean;
}

const statusConfig: Record<string, { color: string; icon: any; label: string }> = {
  healthy: { color: 'badge-success', icon: CheckCircle2, label: 'Healthy' },
  active: { color: 'badge-success', icon: Activity, label: 'Active' },
  completed: { color: 'badge-success', icon: CheckCircle2, label: 'Completed' },
  running: { color: 'badge-primary', icon: Loader2, label: 'Running' },
  processing: { color: 'badge-primary', icon: Loader2, label: 'Processing' },
  pending: { color: 'badge-info', icon: Clock, label: 'Pending' },
  queued: { color: 'badge-info', icon: Clock, label: 'Queued' },
  idle: { color: 'badge-info', icon: Clock, label: 'Idle' },
  warning: { color: 'badge-warning', icon: AlertCircle, label: 'Warning' },
  degraded: { color: 'badge-warning', icon: AlertCircle, label: 'Degraded' },
  retrying: { color: 'badge-warning', icon: RefreshCw, label: 'Retrying' },
  failed: { color: 'badge-danger', icon: XCircle, label: 'Failed' },
  error: { color: 'badge-danger', icon: XCircle, label: 'Error' },
  critical: { color: 'badge-danger', icon: AlertCircle, label: 'Critical' },
  disabled: { color: 'badge-danger', icon: Ban, label: 'Disabled' },
  cancelled: { color: 'badge-danger', icon: Ban, label: 'Cancelled' },
};

export function StatusBadge({ status, size = 'sm', pulse }: StatusBadgeProps) {
  const config = statusConfig[status.toLowerCase()] || { color: 'badge-info', icon: Info, label: status };
  const Icon = config.icon;
  const spinning = status === 'running' || status === 'processing' || status === 'retrying';

  return (
    <span className={`badge ${config.color} ${size === 'md' ? 'px-3 py-1 text-sm' : ''}`}>
      {pulse && (
        <span className={`w-1.5 h-1.5 rounded-full ${config.color.replace('badge-', 'status-dot-')}`} />
      )}
      <Icon className={`w-3 h-3 ${spinning ? 'animate-spin' : ''}`} />
      {config.label}
    </span>
  );
}

// ---- Progress Bar ----

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  color?: 'brand' | 'cyan' | 'emerald' | 'amber' | 'red';
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
}

export function ProgressBar({ value, max = 100, label, showValue, color = 'brand', size = 'md', animated }: ProgressBarProps) {
  const pct = Math.min((value / max) * 100, 100);
  const heights = { sm: 'h-1', md: 'h-2', lg: 'h-3' };
  const barColors = {
    brand: 'bg-indigo-500',
    cyan: 'bg-cyan-500',
    emerald: 'bg-emerald-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
  };

  // Auto-color based on percentage
  const autoColor = pct >= 90 ? 'bg-red-500' : pct >= 75 ? 'bg-amber-500' : barColors[color];

  return (
    <div>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-sm text-zinc-400">{label}</span>}
          {showValue && (
            <span className="text-sm font-medium text-zinc-300">{pct.toFixed(0)}%</span>
          )}
        </div>
      )}
      <div className={`w-full bg-surface-3 rounded-full ${heights[size]} overflow-hidden`}>
        <div
          className={`${autoColor} ${heights[size]} rounded-full transition-all duration-700 ease-out ${animated ? 'animate-pulse' : ''}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---- Empty State ----

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      {icon && (
        <div className="w-16 h-16 rounded-2xl bg-surface-3 flex items-center justify-center text-zinc-500 mb-4">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-sm text-zinc-400 max-w-md mb-6">{description}</p>
      {action}
    </div>
  );
}

// ---- Loading Skeleton ----

export function SkeletonCard() {
  return (
    <div className="premium-card space-y-3">
      <div className="skeleton h-4 w-24" />
      <div className="skeleton h-8 w-32" />
      <div className="skeleton h-3 w-20" />
    </div>
  );
}

export function SkeletonList({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 p-4">
          <div className="skeleton w-10 h-10 rounded-lg" />
          <div className="flex-1 space-y-2">
            <div className="skeleton h-4 w-48" />
            <div className="skeleton h-3 w-32" />
          </div>
          <div className="skeleton h-6 w-20 rounded-full" />
        </div>
      ))}
    </div>
  );
}

// ---- Page Header ----

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  breadcrumb?: { label: string; href?: string }[];
}

export function PageHeader({ title, description, actions, breadcrumb }: PageHeaderProps) {
  return (
    <div className="mb-8">
      {breadcrumb && (
        <nav className="flex items-center gap-2 mb-3 text-sm">
          {breadcrumb.map((item, idx) => (
            <span key={idx} className="flex items-center gap-2">
              {idx > 0 && <span className="text-zinc-600">/</span>}
              <span className={item.href ? 'text-zinc-400 hover:text-white cursor-pointer' : 'text-zinc-500'}>
                {item.label}
              </span>
            </span>
          ))}
        </nav>
      )}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight">{title}</h1>
          {description && <p className="mt-1.5 text-zinc-400 text-sm lg:text-base">{description}</p>}
        </div>
        {actions && <div className="flex items-center gap-3 flex-shrink-0">{actions}</div>}
      </div>
    </div>
  );
}

// ---- Tabs Component ----

interface Tab {
  id: string;
  label: string;
  count?: number;
  icon?: ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (tabId: string) => void;
}

export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
  return (
    <div className="flex items-center gap-1 p-1 bg-surface-2 rounded-xl border border-zinc-800">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
            ${activeTab === tab.id
              ? 'bg-brand-600 text-white shadow-glow-brand'
              : 'text-zinc-400 hover:text-white hover:bg-surface-3'
            }
          `}
        >
          {tab.icon}
          {tab.label}
          {tab.count !== undefined && (
            <span className={`text-xs px-1.5 py-0.5 rounded-full ${
              activeTab === tab.id ? 'bg-white/20' : 'bg-surface-4'
            }`}>
              {tab.count}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// ---- Modal ----

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export function Modal({ open, onClose, title, description, children, footer, size = 'md' }: ModalProps) {
  if (!open) return null;

  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className={`relative w-full ${sizes[size]} bg-surface-2 border border-zinc-800 rounded-2xl shadow-elevation-3 animate-scale-in overflow-hidden`}>
        {/* Header */}
        <div className="px-6 py-4 border-b border-zinc-800">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          {description && <p className="text-sm text-zinc-400 mt-1">{description}</p>}
        </div>
        {/* Body */}
        <div className="px-6 py-5 max-h-[60vh] overflow-y-auto">
          {children}
        </div>
        {/* Footer */}
        {footer && (
          <div className="px-6 py-4 border-t border-zinc-800 flex items-center justify-end gap-3">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Tooltip ----

interface TooltipProps {
  text: string;
  children: ReactNode;
}

export function Tooltip({ text, children }: TooltipProps) {
  return (
    <div className="relative group">
      {children}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-surface-4 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 shadow-elevation-2">
        {text}
        <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 bg-surface-4 rotate-45" />
      </div>
    </div>
  );
}

// ---- Avatar ----

interface AvatarProps {
  name: string;
  src?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function Avatar({ name, src, size = 'md' }: AvatarProps) {
  const sizes = { sm: 'w-7 h-7 text-xs', md: 'w-9 h-9 text-sm', lg: 'w-12 h-12 text-base' };
  const initials = name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();

  if (src) {
    return <img src={src} alt={name} className={`${sizes[size]} rounded-full object-cover`} />;
  }

  return (
    <div className={`${sizes[size]} rounded-full bg-gradient-brand flex items-center justify-center font-semibold text-white`}>
      {initials}
    </div>
  );
}
