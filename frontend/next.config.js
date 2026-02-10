/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    API_GATEWAY_URL: process.env.API_GATEWAY_URL || 'http://localhost:8000',
    EXPLAINABILITY_URL: process.env.EXPLAINABILITY_URL || 'http://localhost:8008',
    OBSERVABILITY_URL: process.env.OBSERVABILITY_URL || 'http://localhost:8009',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_GATEWAY_URL || 'http://localhost:8000'}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
