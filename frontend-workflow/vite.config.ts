import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:9111',
        changeOrigin: true,
      },
    },
  },
})