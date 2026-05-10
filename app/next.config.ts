import type { NextConfig } from 'next';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH !== undefined
  ? process.env.NEXT_PUBLIC_BASE_PATH
  : '/uk/uk-salary-sacrifice-tool';


const nextConfig: NextConfig = {
  ...(basePath ? { basePath } : {}),
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  // recharts ships CJS interop and relies on browser globals; transpiling
  // it through Next's pipeline avoids ESM/CJS mismatch errors during build.
  transpilePackages: ['recharts'],
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
