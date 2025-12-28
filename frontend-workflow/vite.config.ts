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
        target: 'http://dcai-paper2any-back.nas.cpolar.cn/',  // FastAPI 后端地址
        changeOrigin: true,
      },
    },
  },
})