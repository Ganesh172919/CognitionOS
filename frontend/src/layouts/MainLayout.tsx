// ============================================
// CognitionOS - Main Application Layout
// ============================================

import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import {
  LayoutDashboard, Bot, Code2, ListTodo, GitBranch, Puzzle,
  BarChart3, CreditCard, Settings, Bell, Search, ChevronLeft,
  ChevronRight, LogOut, Zap, Menu, X, Crown, User,
} from 'lucide-react';
import { useNotifications, useLocalStorage } from '../hooks/useApi';
import { APP_NAME, MAIN_NAVIGATION } from '../lib/constants';

const ICON_MAP: Record<string, any> = {
  LayoutDashboard, Bot, Code2, ListTodo, GitBranch, Puzzle,
  BarChart3, CreditCard, Settings,
};

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const router = useRouter();
  const [collapsed, setCollapsed] = useLocalStorage('sidebar-collapsed', false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const { data: notifications } = useNotifications();
  const unreadCount = notifications?.filter((n: any) => !n.read)?.length || 0;

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [router.pathname]);

  // Keyboard shortcut for search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(!searchOpen);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [searchOpen]);

  return (
    <div className="flex h-screen overflow-hidden bg-surface-0">
      {/* Mobile Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:relative z-50 flex flex-col h-full
          bg-surface-1 border-r border-zinc-800/50
          transition-all duration-300 ease-in-out
          ${collapsed ? 'w-[72px]' : 'w-64'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-zinc-800/50">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow-brand">
              <Zap className="w-5 h-5 text-white" />
            </div>
            {!collapsed && (
              <span className="text-lg font-bold text-white truncate animate-fade-in">
                {APP_NAME}
              </span>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {MAIN_NAVIGATION.map((item) => {
            const Icon = ICON_MAP[item.icon] || LayoutDashboard;
            const isActive = router.pathname === item.href ||
              (item.href !== '/' && router.pathname.startsWith(item.href));

            return (
              <Link key={item.id} href={item.href}>
                <div
                  className={`
                    nav-item group relative
                    ${isActive ? 'active' : ''}
                    ${collapsed ? 'justify-center px-0' : ''}
                  `}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-white' : ''}`} />
                  {!collapsed && (
                    <span className="truncate">{item.label}</span>
                  )}
                  {!collapsed && item.tier && (
                    <Crown className="w-3.5 h-3.5 text-amber-400 ml-auto flex-shrink-0" />
                  )}
                  {collapsed && isActive && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-white rounded-r" />
                  )}
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="border-t border-zinc-800/50 p-3">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center gap-2 p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-surface-3 transition-colors"
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
            {!collapsed && <span className="text-sm">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="flex items-center justify-between h-16 px-4 lg:px-6 border-b border-zinc-800/50 bg-surface-1/80 backdrop-blur-xl">
          {/* Left: Mobile Menu + Search */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMobileOpen(true)}
              className="lg:hidden p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-surface-3"
            >
              <Menu className="w-5 h-5" />
            </button>

            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-3 px-4 py-2 rounded-lg bg-surface-2 border border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-400 transition-all w-64 lg:w-80"
            >
              <Search className="w-4 h-4" />
              <span className="text-sm">Search...</span>
              <kbd className="ml-auto text-2xs bg-surface-3 px-1.5 py-0.5 rounded text-zinc-500 font-mono hidden sm:inline">
                ⌘K
              </kbd>
            </button>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2">
            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setNotifOpen(!notifOpen)}
                className="relative p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-surface-3 transition-colors"
              >
                <Bell className="w-5 h-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-red-500 text-white text-2xs flex items-center justify-center font-medium">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                )}
              </button>

              {/* Notification Dropdown */}
              {notifOpen && (
                <div className="absolute right-0 top-12 w-80 bg-surface-2 border border-zinc-800 rounded-xl shadow-elevation-3 z-50 animate-scale-in overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
                    <h3 className="text-sm font-semibold text-white">Notifications</h3>
                    <span className="badge badge-primary">{unreadCount} new</span>
                  </div>
                  <div className="max-h-80 overflow-y-auto">
                    {notifications && notifications.length > 0 ? (
                      notifications.slice(0, 5).map((notif: any) => (
                        <div
                          key={notif.id}
                          className={`px-4 py-3 border-b border-zinc-800/50 hover:bg-surface-3 transition-colors cursor-pointer ${
                            !notif.read ? 'bg-brand-500/5' : ''
                          }`}
                        >
                          <p className="text-sm text-white">{notif.title}</p>
                          <p className="text-xs text-zinc-400 mt-0.5">{notif.message}</p>
                        </div>
                      ))
                    ) : (
                      <div className="px-4 py-8 text-center text-zinc-500 text-sm">
                        No notifications
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* User Avatar */}
            <div className="flex items-center gap-3 pl-3 ml-2 border-l border-zinc-800">
              <div className="w-8 h-8 rounded-full bg-gradient-brand flex items-center justify-center">
                <User className="w-4 h-4 text-white" />
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto bg-surface-0">
          {children}
        </main>
      </div>

      {/* Command Palette / Search Modal */}
      {searchOpen && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setSearchOpen(false)} />
          <div className="relative w-full max-w-xl bg-surface-2 border border-zinc-700 rounded-2xl shadow-elevation-3 animate-scale-in overflow-hidden">
            <div className="flex items-center gap-3 px-4 py-4 border-b border-zinc-800">
              <Search className="w-5 h-5 text-zinc-400" />
              <input
                autoFocus
                type="text"
                placeholder="Search tasks, agents, workflows..."
                className="flex-1 bg-transparent text-white outline-none text-base placeholder-zinc-500"
              />
              <button onClick={() => setSearchOpen(false)} className="p-1 rounded text-zinc-500 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="px-2 py-2">
              <p className="px-3 py-2 text-xs font-medium text-zinc-500 uppercase tracking-wider">Quick Actions</p>
              {[
                { label: 'New Task', icon: ListTodo, href: '/tasks' },
                { label: 'Generate Code', icon: Code2, href: '/code-studio' },
                { label: 'View Analytics', icon: BarChart3, href: '/analytics' },
                { label: 'Manage Agents', icon: Bot, href: '/agents' },
              ].map((action) => (
                <Link key={action.label} href={action.href}>
                  <div
                    onClick={() => setSearchOpen(false)}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-zinc-300 hover:text-white hover:bg-surface-3 cursor-pointer transition-colors"
                  >
                    <action.icon className="w-4 h-4 text-zinc-400" />
                    <span className="text-sm">{action.label}</span>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
