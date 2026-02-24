import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { viteSingleFile } from "vite-plugin-singlefile"
import wasm from "vite-plugin-wasm"
import topLevelAwait from "vite-plugin-top-level-await"

export default defineConfig({
  plugins: [react(), wasm(), topLevelAwait(), viteSingleFile()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@nietzsche/perspektive": path.resolve(__dirname, "../../perspektive.js/src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8080",
      "/metrics": "http://localhost:8080",
    },
  },
  build: {
    // Inline everything — produces a single index.html with no external assets
    assetsInlineLimit: 100_000_000,
    cssCodeSplit: false,
    target: "esnext",
  },
})
