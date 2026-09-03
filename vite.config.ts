import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2022',
    // The worker is the only chunk we care about splitting; everything else is
    // small enough that a single bundle beats the extra round trips.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
        },
      },
    },
  },
  worker: {
    format: 'es',
  },
})
