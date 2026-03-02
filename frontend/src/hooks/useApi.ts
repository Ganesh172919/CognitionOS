// ============================================
// CognitionOS - Custom React Hooks
// ============================================

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from '../lib/api-client';
import type { DashboardData, Task, Agent, Notification } from '../lib/types';

// ---- Dashboard Hook ----

export function useDashboard() {
  return useQuery<DashboardData>({
    queryKey: ['dashboard'],
    queryFn: () => apiClient.getDashboard(),
    refetchInterval: 5000,
  });
}

// ---- Tasks Hooks ----

export function useActiveTasks() {
  return useQuery<Task[]>({
    queryKey: ['active-tasks'],
    queryFn: () => apiClient.getActiveTasks(),
    refetchInterval: 3000,
  });
}

export function useAllTasks(params?: { status?: string; page?: number; limit?: number }) {
  return useQuery({
    queryKey: ['tasks', params],
    queryFn: () => apiClient.getAllTasks(params),
  });
}

export function useTask(taskId: string | null) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: () => apiClient.getTask(taskId!),
    enabled: !!taskId,
  });
}

export function useCreateTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (task: { title: string; description: string; priority?: string }) =>
      apiClient.createTask(task),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      queryClient.invalidateQueries({ queryKey: ['active-tasks'] });
    },
  });
}

// ---- Agents Hook ----

export function useAgents() {
  return useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: () => apiClient.getAgents(),
    refetchInterval: 10000,
  });
}

// ---- Explainability Hook ----

export function useTaskExplanation(taskId: string | null) {
  return useQuery({
    queryKey: ['task-explanation', taskId],
    queryFn: () => apiClient.explainTask(taskId!),
    enabled: !!taskId,
  });
}

// ---- Metrics Hook ----

export function useMetrics(params?: { period?: string; service?: string }) {
  return useQuery({
    queryKey: ['metrics', params],
    queryFn: () => apiClient.getMetrics(params),
    refetchInterval: 15000,
  });
}

export function usePerformanceTrends(params?: { period?: string; granularity?: string }) {
  return useQuery({
    queryKey: ['performance-trends', params],
    queryFn: () => apiClient.getPerformanceTrends(params),
    refetchInterval: 30000,
  });
}

// ---- Billing Hooks ----

export function useSubscription() {
  return useQuery({
    queryKey: ['subscription'],
    queryFn: () => apiClient.getSubscription(),
  });
}

export function useInvoices() {
  return useQuery({
    queryKey: ['invoices'],
    queryFn: () => apiClient.getInvoices(),
  });
}

// ---- API Keys Hook ----

export function useApiKeys() {
  return useQuery({
    queryKey: ['api-keys'],
    queryFn: () => apiClient.getApiKeys(),
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: { name: string; permissions: string[]; expires_in_days?: number }) =>
      apiClient.createApiKey(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });
}

// ---- Notifications Hook ----

export function useNotifications() {
  return useQuery<Notification[]>({
    queryKey: ['notifications'],
    queryFn: () => apiClient.getNotifications(),
    refetchInterval: 15000,
  });
}

// ---- Plugins Hook ----

export function usePlugins(params?: { category?: string; status?: string }) {
  return useQuery({
    queryKey: ['plugins', params],
    queryFn: () => apiClient.getPlugins(params),
  });
}

// ---- Workflows Hook ----

export function useWorkflows() {
  return useQuery({
    queryKey: ['workflows'],
    queryFn: () => apiClient.getWorkflows(),
  });
}

// ---- Code Generation Hook ----

export function useCodeGeneration() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: {
      prompt: string;
      language: string;
      framework?: string;
      context_files?: string[];
      requirements?: string[];
    }) => apiClient.generateCode(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['generation-history'] });
    },
  });
}

export function useGenerationHistory(params?: { page?: number; limit?: number }) {
  return useQuery({
    queryKey: ['generation-history', params],
    queryFn: () => apiClient.getGenerationHistory(params),
  });
}

// ---- UI State Hooks ----

export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initialValue;
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  };

  return [storedValue, setValue] as const;
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    setMatches(media.matches);
    const listener = (e: MediaQueryListEvent) => setMatches(e.matches);
    media.addEventListener('change', listener);
    return () => media.removeEventListener('change', listener);
  }, [query]);

  return matches;
}

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

export function useClickOutside(ref: React.RefObject<HTMLElement>, handler: () => void) {
  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) return;
      handler();
    };
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);
    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}
