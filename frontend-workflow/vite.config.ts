import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'https://dcai-paper2any-back.cpolar.top/',  // FastAPI 后端地址
        changeOrigin: true,
      },
    },
  },
})
