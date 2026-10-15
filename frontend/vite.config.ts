import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          const path = id.replaceAll('\\', '/')
          if (path.includes('/node_modules/zrender/')) return 'zrender'
          if (path.includes('/node_modules/echarts/charts/')) return 'echarts-charts'
          if (path.includes('/node_modules/echarts/components/')) return 'echarts-components'
          if (path.includes('/node_modules/echarts/core')) return 'echarts-core'
        },
      },
    },
  },
  server: {
    proxy: {
      '/api/v1': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
