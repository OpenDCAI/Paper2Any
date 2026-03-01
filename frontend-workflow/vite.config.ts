import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3123,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8123',
        changeOrigin: true,
      },
      '/outputs': {
        target: 'http://localhost:8123',
        changeOrigin: true,
      },
    },
  },
})

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 3111,
//     open: true,
//     allowedHosts: true,
//     proxy: {
//       '/api': {
//         // target: 'http://localhost:8000',
//         target: 'http://paper2any-test-back.nas.cpolar.cn/',  // FastAPI 后端地址
//         changeOrigin: true,
//       },
//       '/outputs': {
//         // target: 'http://localhost:8000',
//         target: 'http://paper2any-test-back.nas.cpolar.cn/',
//         changeOrigin: true,
//       },
//     },
//   },
// })