import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // recharts ships CJS interop and relies on browser globals; transpiling
  // it through Next's pipeline avoids ESM/CJS mismatch errors during build.
  transpilePackages: ['recharts'],
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
