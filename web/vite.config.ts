import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// GitHub Pages serves the site from /<repo>/; override with VITE_BASE for other hosts.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/Connect-4-Game/",
  plugins: [react()],
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm"] },
  server: { watch: { usePolling: true, interval: 300 } },
  worker: { format: "es" },
  build: { target: "es2022" },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    testTimeout: 60_000,
  },
});
