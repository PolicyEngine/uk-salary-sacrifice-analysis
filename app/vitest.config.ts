import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.js'],
    server: {
      deps: {
        // Inline @policyengine/ui-kit so jsdom can resolve its ESM exports
        // through the same pipeline as the rest of the PolicyEngine portfolio.
        inline: ['@policyengine/ui-kit'],
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
