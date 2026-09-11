import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.RL_BACKEND ?? "http://localhost:8901";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  define: {
    "import.meta.env.VITE_PUBLIC_DEMO": JSON.stringify(
      mode === "public" ? "1" : process.env.VITE_PUBLIC_DEMO ?? "0",
    ),
  },
  server: {
    port: 5180,
    proxy: {
      "/api": backend,
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
}));
