// ============================================
// CognitionOS - Settings Page
// ============================================

import { useState } from 'react';
import {
  Settings, User, Key, Bell, Shield, Globe, Palette, Save,
  Plus, Eye, EyeOff, Trash2, Copy, Check, AlertCircle,
  Monitor, Moon, Sun, Lock, Mail, Database, Code2,
} from 'lucide-react';
import { PageHeader, Tabs, Modal, StatusBadge } from '../components/ui';
import { formatRelativeTime, formatDate } from '../lib/utils';

const demoApiKeys = [
  { id: '1', name: 'Production API', key_prefix: 'cog_prod_xK9...', created_at: '2026-01-15T10:00:00Z', last_used_at: '2026-03-01T06:30:00Z', status: 'active', permissions: ['read', 'write', 'admin'] },
  { id: '2', name: 'Development', key_prefix: 'cog_dev_mP2...', created_at: '2026-02-01T10:00:00Z', last_used_at: '2026-02-28T18:00:00Z', status: 'active', permissions: ['read', 'write'] },
  { id: '3', name: 'CI/CD Pipeline', key_prefix: 'cog_ci_jR7...', created_at: '2026-02-15T10:00:00Z', last_used_at: '2026-03-01T02:00:00Z', status: 'active', permissions: ['read'] },
  { id: '4', name: 'Deprecated Key', key_prefix: 'cog_old_dF5...', created_at: '2025-12-01T10:00:00Z', status: 'revoked', permissions: ['read', 'write'] },
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('profile');
  const [showCreateKey, setShowCreateKey] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [showNewKey, setShowNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Profile form state
  const [profile, setProfile] = useState({
    name: 'Alex Developer',
    email: 'alex@cognitionos.ai',
    organization: 'CognitionOS Inc.',
    role: 'admin',
  });

  // Preferences state
  const [prefs, setPrefs] = useState({
    theme: 'dark',
    notifications: true,
    emailDigest: 'daily',
    defaultModel: 'gemini-2.0-flash',
    defaultLanguage: 'typescript',
  });

  const handleCopyKey = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Settings"
        description="Manage your account, API keys, and platform preferences"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Settings Sidebar */}
        <div className="lg:w-56 flex-shrink-0">
          <nav className="space-y-1">
            {[
              { id: 'profile', label: 'Profile', icon: User },
              { id: 'api-keys', label: 'API Keys', icon: Key },
              { id: 'preferences', label: 'Preferences', icon: Palette },
              { id: 'notifications', label: 'Notifications', icon: Bell },
              { id: 'security', label: 'Security', icon: Shield },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full nav-item ${activeTab === item.id ? 'active' : ''}`}
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Settings Content */}
        <div className="flex-1 min-w-0">
          {/* Profile */}
          {activeTab === 'profile' && (
            <div className="premium-card space-y-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <User className="w-5 h-5 text-brand-400" /> Profile Settings
              </h2>
              <div className="flex items-center gap-6 pb-6 border-b border-zinc-800">
                <div className="w-20 h-20 rounded-2xl bg-gradient-brand flex items-center justify-center text-2xl font-bold text-white shadow-glow-brand">
                  {profile.name.split(' ').map(w => w[0]).join('')}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">{profile.name}</h3>
                  <p className="text-zinc-400">{profile.email}</p>
                  <span className="badge badge-primary mt-2">{profile.role}</span>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-zinc-300 mb-1.5 block">Full Name</label>
                  <input
                    type="text"
                    value={profile.name}
                    onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                    className="input-base"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-300 mb-1.5 block">Email</label>
                  <input
                    type="email"
                    value={profile.email}
                    onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                    className="input-base"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-300 mb-1.5 block">Organization</label>
                  <input
                    type="text"
                    value={profile.organization}
                    onChange={(e) => setProfile({ ...profile, organization: e.target.value })}
                    className="input-base"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-300 mb-1.5 block">Role</label>
                  <select value={profile.role} onChange={(e) => setProfile({ ...profile, role: e.target.value })} className="input-base">
                    <option value="admin">Admin</option>
                    <option value="developer">Developer</option>
                    <option value="viewer">Viewer</option>
                    <option value="billing_admin">Billing Admin</option>
                  </select>
                </div>
              </div>
              <div className="flex justify-end">
                <button className="btn btn-primary btn-md">
                  <Save className="w-4 h-4" /> Save Changes
                </button>
              </div>
            </div>
          )}

          {/* API Keys */}
          {activeTab === 'api-keys' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Key className="w-5 h-5 text-brand-400" /> API Keys
                  </h2>
                  <p className="text-sm text-zinc-400 mt-1">Manage your API keys for programmatic access</p>
                </div>
                <button onClick={() => setShowCreateKey(true)} className="btn btn-primary btn-md">
                  <Plus className="w-4 h-4" /> Create Key
                </button>
              </div>

              <div className="premium-card p-0 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800 text-left">
                      <th className="px-6 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Name</th>
                      <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Key</th>
                      <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Permissions</th>
                      <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Status</th>
                      <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider">Last Used</th>
                      <th className="px-4 py-4 text-xs font-medium text-zinc-400 uppercase tracking-wider"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50">
                    {demoApiKeys.map((key) => (
                      <tr key={key.id} className="table-row">
                        <td className="px-6 py-4 text-white font-medium">{key.name}</td>
                        <td className="px-4 py-4">
                          <code className="text-xs text-zinc-400 bg-surface-3 px-2 py-1 rounded">{key.key_prefix}</code>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex gap-1">
                            {key.permissions.map((p) => (
                              <span key={p} className="badge badge-info text-2xs">{p}</span>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-4"><StatusBadge status={key.status} /></td>
                        <td className="px-4 py-4 text-zinc-400 text-xs">
                          {key.last_used_at ? formatRelativeTime(key.last_used_at) : 'Never'}
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex gap-1">
                            <button className="btn btn-ghost btn-sm" onClick={() => handleCopyKey(key.key_prefix)}>
                              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                            </button>
                            <button className="btn btn-ghost btn-sm text-red-400 hover:text-red-300">
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Warning */}
              <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
                <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-amber-300 font-medium">API Key Security</p>
                  <p className="text-xs text-zinc-400 mt-1">Keep your API keys secure. Do not share them in public repositories or client-side code. Rotate keys periodically for security.</p>
                </div>
              </div>
            </div>
          )}

          {/* Preferences */}
          {activeTab === 'preferences' && (
            <div className="premium-card space-y-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Palette className="w-5 h-5 text-brand-400" /> Preferences
              </h2>

              <div className="space-y-6">
                <div>
                  <label className="text-sm text-zinc-300 mb-3 block font-medium">Theme</label>
                  <div className="flex gap-3">
                    {[
                      { value: 'dark', icon: Moon, label: 'Dark' },
                      { value: 'light', icon: Sun, label: 'Light' },
                      { value: 'system', icon: Monitor, label: 'System' },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setPrefs({ ...prefs, theme: option.value })}
                        className={`flex items-center gap-2 px-4 py-3 rounded-xl transition-all ${
                          prefs.theme === option.value
                            ? 'bg-brand-600 text-white shadow-glow-brand'
                            : 'bg-surface-3 text-zinc-400 border border-zinc-800 hover:border-zinc-700'
                        }`}
                      >
                        <option.icon className="w-4 h-4" />
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-zinc-300 mb-1.5 block">Default AI Model</label>
                    <select value={prefs.defaultModel} onChange={(e) => setPrefs({ ...prefs, defaultModel: e.target.value })} className="input-base">
                      <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                      <option value="gemini-2.0-pro">Gemini 2.0 Pro</option>
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                      <option value="claude-3.5-sonnet">Claude 3.5 Sonnet</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-zinc-300 mb-1.5 block">Default Language</label>
                    <select value={prefs.defaultLanguage} onChange={(e) => setPrefs({ ...prefs, defaultLanguage: e.target.value })} className="input-base">
                      <option value="typescript">TypeScript</option>
                      <option value="python">Python</option>
                      <option value="javascript">JavaScript</option>
                      <option value="rust">Rust</option>
                      <option value="go">Go</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-zinc-300 mb-1.5 block">Email Digest</label>
                    <select value={prefs.emailDigest} onChange={(e) => setPrefs({ ...prefs, emailDigest: e.target.value })} className="input-base">
                      <option value="daily">Daily</option>
                      <option value="weekly">Weekly</option>
                      <option value="never">Never</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="flex justify-end pt-4 border-t border-zinc-800">
                <button className="btn btn-primary btn-md">
                  <Save className="w-4 h-4" /> Save Preferences
                </button>
              </div>
            </div>
          )}

          {/* Notifications */}
          {activeTab === 'notifications' && (
            <div className="premium-card space-y-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Bell className="w-5 h-5 text-brand-400" /> Notification Settings
              </h2>
              <div className="space-y-4">
                {[
                  { label: 'Task Completion', desc: 'Get notified when AI agents complete tasks', enabled: true },
                  { label: 'Task Failures', desc: 'Receive alerts when tasks fail or encounter errors', enabled: true },
                  { label: 'Usage Alerts', desc: 'Warnings when approaching usage limits', enabled: true },
                  { label: 'Billing Updates', desc: 'Payment confirmations and invoice notifications', enabled: true },
                  { label: 'Security Alerts', desc: 'Authentication events and suspicious activity', enabled: true },
                  { label: 'New Features', desc: 'Product updates and new feature announcements', enabled: false },
                  { label: 'Community Updates', desc: 'New plugins, templates, and community content', enabled: false },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between p-4 rounded-xl bg-surface-3/50 border border-zinc-800">
                    <div>
                      <p className="text-sm font-medium text-white">{item.label}</p>
                      <p className="text-xs text-zinc-400 mt-0.5">{item.desc}</p>
                    </div>
                    <button className={`relative w-10 h-5 rounded-full transition-colors ${item.enabled ? 'bg-brand-600' : 'bg-surface-4'}`}>
                      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${item.enabled ? 'left-5.5' : 'left-0.5'}`} style={{ left: item.enabled ? '22px' : '2px' }} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Security */}
          {activeTab === 'security' && (
            <div className="space-y-6">
              <div className="premium-card space-y-6">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Shield className="w-5 h-5 text-brand-400" /> Security Settings
                </h2>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-zinc-300 mb-1.5 block">Change Password</label>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <input type="password" placeholder="Current password" className="input-base" />
                      <input type="password" placeholder="New password" className="input-base" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-surface-3/50 border border-zinc-800">
                    <div>
                      <p className="text-sm font-medium text-white">Two-Factor Authentication</p>
                      <p className="text-xs text-zinc-400 mt-0.5">Add an extra layer of security with 2FA</p>
                    </div>
                    <button className="btn btn-secondary btn-sm">
                      <Lock className="w-3.5 h-3.5" /> Enable 2FA
                    </button>
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-surface-3/50 border border-zinc-800">
                    <div>
                      <p className="text-sm font-medium text-white">SSO Configuration</p>
                      <p className="text-xs text-zinc-400 mt-0.5">Connect with SAML or OAuth providers</p>
                    </div>
                    <span className="badge badge-warning">Enterprise</span>
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-surface-3/50 border border-zinc-800">
                    <div>
                      <p className="text-sm font-medium text-white">Session Management</p>
                      <p className="text-xs text-zinc-400 mt-0.5">View and manage active sessions</p>
                    </div>
                    <button className="btn btn-ghost btn-sm text-zinc-400">3 active sessions</button>
                  </div>
                </div>
              </div>

              {/* Danger Zone */}
              <div className="premium-card border-red-500/20">
                <h3 className="text-sm font-semibold text-red-400 mb-4">Danger Zone</h3>
                <div className="flex items-center justify-between p-4 rounded-xl bg-red-500/5 border border-red-500/20">
                  <div>
                    <p className="text-sm font-medium text-white">Delete Account</p>
                    <p className="text-xs text-zinc-400 mt-0.5">Permanently delete your account and all associated data</p>
                  </div>
                  <button className="btn btn-danger btn-sm">
                    <Trash2 className="w-3.5 h-3.5" /> Delete Account
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create API Key Modal */}
      <Modal
        open={showCreateKey}
        onClose={() => setShowCreateKey(false)}
        title="Create API Key"
        description="Generate a new API key for programmatic access"
        footer={
          <>
            <button className="btn btn-secondary btn-md" onClick={() => setShowCreateKey(false)}>Cancel</button>
            <button className="btn btn-primary btn-md" disabled={!newKeyName.trim()}>
              <Key className="w-4 h-4" /> Generate Key
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Key Name</label>
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="e.g., Production API"
              className="input-base"
            />
          </div>
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Permissions</label>
            <div className="flex gap-2">
              {['read', 'write', 'admin'].map((p) => (
                <button key={p} className="px-3 py-1.5 rounded-lg text-sm bg-surface-3 text-zinc-400 border border-zinc-800 hover:border-zinc-700 capitalize">
                  {p}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-sm text-zinc-300 mb-1.5 block">Expiration</label>
            <select className="input-base">
              <option value="30">30 days</option>
              <option value="90">90 days</option>
              <option value="365">1 year</option>
              <option value="0">Never expires</option>
            </select>
          </div>
        </div>
      </Modal>
    </div>
  );
}
