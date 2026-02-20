import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { viteSingleFile } from "vite-plugin-singlefile"

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Inline everything — produces a single index.html with no external assets
    assetsInlineLimit: 100_000_000,
    cssCodeSplit: false,
    target: "esnext",
  },
})
