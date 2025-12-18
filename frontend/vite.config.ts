import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const timestamp = Date.now();

export default defineConfig(() => {
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
          // Proxy API requests to backend
          '/api': {
            target: 'http://localhost:8000',
            changeOrigin: true,
          },
          '/generated': {
            target: 'http://localhost:8000',
            changeOrigin: true,
          },
        },
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      build: {
        rollupOptions: {
          output: {
            entryFileNames: `[name].js?t=${timestamp}`,
            chunkFileNames: `[name].js?t=${timestamp}`,
            assetFileNames: `[name].[ext]?t=${timestamp}`,
          },
        },
      },
    };
});
