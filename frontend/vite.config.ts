import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (uvicorn lnp_rag.api.main:app) is proxied under /api, so the
// browser never makes a cross-origin request. Override with API_URL=http://host:port.
const api = {
  "/api": {
    target: process.env.API_URL ?? "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
    timeout: 0, // agent searches can take several minutes
    proxyTimeout: 0,
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: api },
  preview: { port: 4173, proxy: api },
});
