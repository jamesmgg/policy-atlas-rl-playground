import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.RL_BACKEND ?? "http://localhost:8901";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: {
      "/api": backend,
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
});
