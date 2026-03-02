// ============================================
// CognitionOS - Billing & Subscription Page
// ============================================

import { useState } from 'react';
import {
  CreditCard, Check, Crown, Zap, ArrowRight, Download,
  Shield, Users, Globe, Clock, AlertCircle, ExternalLink,
  Star, TrendingUp, Sparkles,
} from 'lucide-react';
import { PageHeader, Tabs, ProgressBar, StatCard, Modal } from '../components/ui';
import { formatCurrency, formatNumber, formatDate, formatPercent } from '../lib/utils';
import { SUBSCRIPTION_TIERS } from '../lib/constants';

const demoUsage = {
  tokens: { used: 234567, limit: 500000 },
  apiCalls: { used: 4523, limit: 10000 },
  agents: { used: 5, limit: -1 },
  storage: { used: 2.4, limit: 10 },
};

const demoInvoices = [
  { id: 'INV-2026-003', period: 'March 2026', amount: 49.00, status: 'pending', date: '2026-03-01' },
  { id: 'INV-2026-002', period: 'February 2026', amount: 49.00, status: 'paid', date: '2026-02-01' },
  { id: 'INV-2026-001', period: 'January 2026', amount: 49.00, status: 'paid', date: '2026-01-01' },
  { id: 'INV-2025-012', period: 'December 2025', amount: 49.00, status: 'paid', date: '2025-12-01' },
];

export default function BillingPage() {
  const [activeTab, setActiveTab] = useState('plan');
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly');
  const [showUpgrade, setShowUpgrade] = useState(false);
  const currentTier = 'pro';

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Billing & Subscription"
        description="Manage your plan, usage, and payment methods"
      />

      <Tabs
        tabs={[
          { id: 'plan', label: 'Current Plan', icon: <Crown className="w-4 h-4" /> },
          { id: 'usage', label: 'Usage', icon: <Zap className="w-4 h-4" /> },
          { id: 'invoices', label: 'Invoices', icon: <CreditCard className="w-4 h-4" /> },
        ]}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {activeTab === 'plan' && (
        <div className="space-y-6">
          {/* Current Plan Banner */}
          <div className="relative overflow-hidden rounded-2xl border border-brand-500/30 bg-gradient-to-r from-brand-600/10 via-violet-600/10 to-cyan-600/10 p-8">
            <div className="absolute top-0 right-0 w-64 h-64 bg-brand-500/5 rounded-full blur-3xl" />
            <div className="relative flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <Crown className="w-6 h-6 text-amber-400" />
                  <span className="badge badge-primary text-sm px-3 py-1">Pro Plan</span>
                </div>
                <h2 className="text-2xl font-bold text-white">$49<span className="text-lg text-zinc-400">/month</span></h2>
                <p className="text-zinc-400 mt-1">Your plan renews on March 31, 2026</p>
              </div>
              <div className="flex gap-3">
                <button onClick={() => setShowUpgrade(true)} className="btn btn-primary btn-md">
                  <TrendingUp className="w-4 h-4" /> Upgrade to Enterprise
                </button>
                <button className="btn btn-secondary btn-md">Manage Payment</button>
              </div>
            </div>
          </div>

          {/* Plan Comparison */}
          <div className="flex items-center justify-center gap-3 mb-6">
            <span className="text-sm text-zinc-400">Monthly</span>
            <button
              onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'annual' : 'monthly')}
              className={`relative w-12 h-6 rounded-full transition-colors ${billingCycle === 'annual' ? 'bg-brand-600' : 'bg-surface-4'}`}
            >
              <span className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${billingCycle === 'annual' ? 'left-7' : 'left-1'}`} />
            </button>
            <span className="text-sm text-zinc-400">Annual</span>
            <span className="badge badge-success text-xs">Save 20%</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(SUBSCRIPTION_TIERS).map(([key, tier]) => {
              const price = billingCycle === 'annual' ? tier.annualPrice / 12 : tier.monthlyPrice;
              const isCurrent = key === currentTier;

              return (
                <div
                  key={key}
                  className={`premium-card relative ${
                    tier.highlighted ? 'border-brand-500/50 shadow-glow-brand' : ''
                  } ${isCurrent ? 'ring-2 ring-brand-500' : ''}`}
                >
                  {tier.highlighted && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <span className="bg-gradient-brand text-white text-xs font-bold px-4 py-1 rounded-full shadow-glow-brand">
                        Most Popular
                      </span>
                    </div>
                  )}
                  <div className="p-6 space-y-6">
                    <div>
                      <h3 className="text-xl font-bold text-white">{tier.name}</h3>
                      <div className="mt-3">
                        <span className="text-3xl font-bold text-white">${price.toFixed(0)}</span>
                        <span className="text-zinc-400 text-sm ml-1">/month</span>
                      </div>
                      {billingCycle === 'annual' && tier.monthlyPrice > 0 && (
                        <p className="text-xs text-emerald-400 mt-1">
                          ${tier.annualPrice}/year (save ${(tier.monthlyPrice * 12 - tier.annualPrice).toFixed(0)})
                        </p>
                      )}
                    </div>
                    <ul className="space-y-3">
                      {tier.features.map((feature) => (
                        <li key={feature} className="flex items-start gap-2 text-sm">
                          <Check className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                          <span className="text-zinc-300">{feature}</span>
                        </li>
                      ))}
                    </ul>
                    <button
                      className={`w-full btn btn-md ${
                        isCurrent ? 'btn-secondary cursor-default' : tier.highlighted ? 'btn-primary' : 'btn-secondary'
                      }`}
                      disabled={isCurrent}
                    >
                      {isCurrent ? 'Current Plan' : key === 'free' ? 'Downgrade' : 'Upgrade'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeTab === 'usage' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title="Tokens Used"
              value={`${formatNumber(demoUsage.tokens.used)} / ${formatNumber(demoUsage.tokens.limit)}`}
              icon={<Zap className="w-5 h-5" />}
              color="amber"
            />
            <StatCard
              title="API Calls"
              value={`${formatNumber(demoUsage.apiCalls.used)} / ${formatNumber(demoUsage.apiCalls.limit)}`}
              icon={<Globe className="w-5 h-5" />}
              color="cyan"
            />
            <StatCard
              title="AI Agents"
              value={`${demoUsage.agents.used} / Unlimited`}
              icon={<Sparkles className="w-5 h-5" />}
              color="brand"
            />
            <StatCard
              title="Storage"
              value={`${demoUsage.storage.used}GB / ${demoUsage.storage.limit}GB`}
              icon={<Shield className="w-5 h-5" />}
              color="emerald"
            />
          </div>

          <div className="premium-card space-y-6">
            <h2 className="section-header"><Zap className="w-5 h-5 text-amber-400" /> Usage Breakdown</h2>
            <div className="space-y-6">
              <ProgressBar
                value={demoUsage.tokens.used}
                max={demoUsage.tokens.limit}
                label={`Token Usage (${formatNumber(demoUsage.tokens.used)} / ${formatNumber(demoUsage.tokens.limit)})`}
                showValue
                color="amber"
              />
              <ProgressBar
                value={demoUsage.apiCalls.used}
                max={demoUsage.apiCalls.limit}
                label={`API Calls (${formatNumber(demoUsage.apiCalls.used)} / ${formatNumber(demoUsage.apiCalls.limit)} daily)`}
                showValue
                color="cyan"
              />
              <ProgressBar
                value={demoUsage.storage.used}
                max={demoUsage.storage.limit}
                label={`Storage (${demoUsage.storage.used}GB / ${demoUsage.storage.limit}GB)`}
                showValue
                color="emerald"
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === 'invoices' && (
        <div className="premium-card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left">
                <th className="px-6 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Invoice</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Period</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Amount</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Status</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Date</th>
                <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {demoInvoices.map((invoice) => (
                <tr key={invoice.id} className="table-row">
                  <td className="px-6 py-4 text-white font-mono text-sm">{invoice.id}</td>
                  <td className="px-4 py-4 text-zinc-300">{invoice.period}</td>
                  <td className="px-4 py-4 text-white font-medium">{formatCurrency(invoice.amount)}</td>
                  <td className="px-4 py-4">
                    <span className={`badge ${invoice.status === 'paid' ? 'badge-success' : 'badge-warning'}`}>
                      {invoice.status}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-zinc-400">{formatDate(invoice.date)}</td>
                  <td className="px-4 py-4">
                    <button className="btn btn-ghost btn-sm">
                      <Download className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
