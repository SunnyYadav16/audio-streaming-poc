import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward WebSocket + REST calls to the FastAPI backend
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
      '/rooms': {
        target: 'http://localhost:8000',
      },
      '/recordings': {
        target: 'http://localhost:8000',
      },
    },
  },
})
