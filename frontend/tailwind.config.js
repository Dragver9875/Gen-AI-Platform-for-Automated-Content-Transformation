/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "surface-container-low": "#f8fafc",
        "surface": "#ffffff",
        "on-surface": "#0f172a",
        "surface-bright": "#ffffff",
        "surface-container-highest": "#e2e8f0",
        "primary": "#4f46e5",
        "secondary-container": "#eef2ff",
        "surface-dim": "#f1f5f9",
        "outline": "#cbd5e1",
        "background": "#ffffff",
        "error": "#dc2626",
        "tertiary": "#059669",
        "surface-variant": "#f1f5f9",
        "surface-container": "#f8fafc",
        "on-surface-variant": "#64748b"
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      }
    },
  },
  plugins: [],
}
