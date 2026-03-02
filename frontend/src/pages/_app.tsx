import type { AppProps } from 'next/app';
import Head from 'next/head';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from '../layouts/MainLayout';
import '../styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 2,
      staleTime: 5000,
    },
  },
});

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>CognitionOS - AI-Powered Code Intelligence Platform</title>
        <meta name="description" content="Enterprise-grade AI agent orchestration and autonomous code generation platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <QueryClientProvider client={queryClient}>
        <Layout>
          <Component {...pageProps} />
        </Layout>
      </QueryClientProvider>
    </>
  );
}
