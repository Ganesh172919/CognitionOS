// ============================================
// CognitionOS - Plugins Marketplace Page
// ============================================

import { useState } from 'react';
import {
  Puzzle, Search, Download, Star, Check, ExternalLink,
  Plus, Settings, Trash2, Filter, Grid, List,
  Code2, Database, Shield, Zap, Globe, Brain,
  GitBranch, MessageSquare, BarChart3, Lock,
} from 'lucide-react';
import { PageHeader, Tabs, StatusBadge, EmptyState, Modal } from '../components/ui';
import { formatNumber } from '../lib/utils';

const CATEGORIES = ['All', 'AI Models', 'Code Analysis', 'DevOps', 'Security', 'Data', 'Communication', 'Analytics'];

const categoryIcons: Record<string, any> = {
  'AI Models': Brain,
  'Code Analysis': Code2,
  'DevOps': GitBranch,
  'Security': Shield,
  'Data': Database,
  'Communication': MessageSquare,
  'Analytics': BarChart3,
};

const demoPlugins = [
  { id: '1', name: 'GPT-4 Turbo Connector', description: 'Connect OpenAI GPT-4 Turbo as an alternative generation backend with streaming support', version: '2.1.0', author: 'CognitionOS Team', category: 'AI Models', install_count: 15234, rating: 4.8, status: 'installed', icon: Brain, gradient: 'from-violet-600 to-purple-600' },
  { id: '2', name: 'ESLint Deep Analyzer', description: 'Advanced static code analysis with custom rule engine and auto-fix capabilities', version: '3.0.1', author: 'CodeQuality Labs', category: 'Code Analysis', install_count: 8920, rating: 4.6, status: 'available', icon: Code2, gradient: 'from-indigo-600 to-blue-600' },
  { id: '3', name: 'GitHub Actions Bridge', description: 'Bi-directional integration with GitHub Actions for automated CI/CD workflows', version: '1.5.2', author: 'DevFlow Inc', category: 'DevOps', install_count: 12450, rating: 4.9, status: 'installed', icon: GitBranch, gradient: 'from-emerald-600 to-teal-600' },
  { id: '4', name: 'Vault Security Scanner', description: 'Automated security vulnerability scanning and secret detection in generated code', version: '2.0.0', author: 'SecureCode', category: 'Security', install_count: 6780, rating: 4.7, status: 'available', icon: Shield, gradient: 'from-red-600 to-orange-600' },
  { id: '5', name: 'PostgreSQL Schema Gen', description: 'Intelligent database schema generation with migration scripts and optimization hints', version: '1.8.3', author: 'DataForge', category: 'Data', install_count: 9340, rating: 4.5, status: 'available', icon: Database, gradient: 'from-cyan-600 to-blue-600' },
  { id: '6', name: 'Slack Notifications', description: 'Real-time task status notifications and alerts delivered to your Slack channels', version: '1.2.0', author: 'CognitionOS Team', category: 'Communication', install_count: 18900, rating: 4.4, status: 'installed', icon: MessageSquare, gradient: 'from-amber-600 to-yellow-600' },
  { id: '7', name: 'DataDog Metrics Export', description: 'Export performance metrics and traces to DataDog for enterprise observability', version: '1.0.5', author: 'Observability Plus', category: 'Analytics', install_count: 4560, rating: 4.3, status: 'available', icon: BarChart3, gradient: 'from-pink-600 to-rose-600' },
  { id: '8', name: 'Claude 3.5 Sonnet', description: 'Connect Anthropic Claude 3.5 Sonnet as an alternative reasoning engine', version: '1.1.0', author: 'CognitionOS Team', category: 'AI Models', install_count: 11200, rating: 4.9, status: 'available', icon: Brain, gradient: 'from-orange-600 to-red-600' },
];

export default function PluginsPage() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [activeTab, setActiveTab] = useState('marketplace');
  const [selectedPlugin, setSelectedPlugin] = useState<any>(null);

  const filteredPlugins = demoPlugins.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         p.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || p.category === selectedCategory;
    const matchesTab = activeTab === 'marketplace' || (activeTab === 'installed' && p.status === 'installed');
    return matchesSearch && matchesCategory && matchesTab;
  });

  const installedCount = demoPlugins.filter(p => p.status === 'installed').length;

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Plugins"
        description="Extend CognitionOS with powerful integrations and extensions"
        actions={
          <div className="flex items-center gap-2">
            <button className="btn btn-secondary btn-md">
              <Plus className="w-4 h-4" /> Submit Plugin
            </button>
          </div>
        }
      />

      <Tabs
        tabs={[
          { id: 'marketplace', label: 'Marketplace', count: demoPlugins.length, icon: <Globe className="w-4 h-4" /> },
          { id: 'installed', label: 'Installed', count: installedCount, icon: <Check className="w-4 h-4" /> },
        ]}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Filters */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center gap-4">
        <div className="relative flex-1 w-full lg:max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search plugins..."
            className="input-base pl-10"
          />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedCategory === cat
                  ? 'bg-brand-600 text-white'
                  : 'bg-surface-3 text-zinc-400 hover:text-white border border-zinc-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Plugin Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredPlugins.map((plugin) => {
          const Icon = plugin.icon || Puzzle;
          return (
            <div
              key={plugin.id}
              className="glow-card cursor-pointer group"
              onClick={() => setSelectedPlugin(plugin)}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${plugin.gradient} flex items-center justify-center shadow-lg`}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white group-hover:text-brand-300 transition-colors">{plugin.name}</h3>
                      <p className="text-xs text-zinc-500">v{plugin.version} · {plugin.author}</p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-zinc-400 mb-4 line-clamp-2">{plugin.description}</p>
                <div className="flex items-center justify-between pt-4 border-t border-zinc-800/50">
                  <div className="flex items-center gap-4 text-xs text-zinc-500">
                    <span className="flex items-center gap-1">
                      <Download className="w-3 h-3" /> {formatNumber(plugin.install_count)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Star className="w-3 h-3 text-amber-400" /> {plugin.rating}
                    </span>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); }}
                    className={`btn btn-sm ${
                      plugin.status === 'installed'
                        ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                        : 'btn-primary'
                    }`}
                  >
                    {plugin.status === 'installed' ? (
                      <><Check className="w-3 h-3" /> Installed</>
                    ) : (
                      <><Download className="w-3 h-3" /> Install</>
                    )}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filteredPlugins.length === 0 && (
        <EmptyState
          icon={<Puzzle className="w-8 h-8" />}
          title="No plugins found"
          description="Try adjusting your search or category filters"
        />
      )}

      {/* Plugin Detail Modal */}
      <Modal
        open={!!selectedPlugin}
        onClose={() => setSelectedPlugin(null)}
        title={selectedPlugin?.name || ''}
        description={`v${selectedPlugin?.version} by ${selectedPlugin?.author}`}
        size="md"
        footer={
          <>
            <button className="btn btn-secondary btn-md" onClick={() => setSelectedPlugin(null)}>Close</button>
            <button className={`btn btn-md ${selectedPlugin?.status === 'installed' ? 'btn-danger' : 'btn-primary'}`}>
              {selectedPlugin?.status === 'installed' ? (
                <><Trash2 className="w-4 h-4" /> Uninstall</>
              ) : (
                <><Download className="w-4 h-4" /> Install Plugin</>
              )}
            </button>
          </>
        }
      >
        {selectedPlugin && (
          <div className="space-y-4">
            <p className="text-zinc-300">{selectedPlugin.description}</p>
            <div className="grid grid-cols-3 gap-4">
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Downloads</p>
                <p className="text-lg font-bold text-white">{formatNumber(selectedPlugin.install_count)}</p>
              </div>
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Rating</p>
                <p className="text-lg font-bold text-amber-400">{selectedPlugin.rating}</p>
              </div>
              <div className="glass-card p-3 text-center">
                <p className="text-xs text-zinc-400">Category</p>
                <p className="text-sm font-medium text-white mt-1">{selectedPlugin.category}</p>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
