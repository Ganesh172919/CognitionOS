/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    API_GATEWAY_URL: process.env.API_GATEWAY_URL || 'http://localhost:8000',
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:8100',
    EXPLAINABILITY_URL: process.env.EXPLAINABILITY_URL || 'http://localhost:8008',
    OBSERVABILITY_URL: process.env.OBSERVABILITY_URL || 'http://localhost:8009',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_BASE_URL || 'http://localhost:8100'}/:path*`,
      },
    ];
  },
  images: {
    domains: ['localhost'],
  },
  // Enable SWC minification for faster builds
  swcMinify: true,
  // Optimize fonts
  optimizeFonts: true,
  // Compiler options
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },
};

module.exports = nextConfig;
