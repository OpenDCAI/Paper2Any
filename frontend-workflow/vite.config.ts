import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 3123,
//     open: true,
//     allowedHosts: true,
//     proxy: {
//       '/api': {
//         target: 'http://localhost:8123',
//         changeOrigin: true,
//       },
//       '/outputs': {
//         target: 'http://localhost:8123',
//         changeOrigin: true,
//       },
//     },
//   },
// })

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/outputs': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/onlyoffice': {
        target: 'http://localhost:8082',
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/onlyoffice/, ''),
      },
    },
  },
})
