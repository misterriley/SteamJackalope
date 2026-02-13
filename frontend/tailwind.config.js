/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0f172a",
        foreground: "#f8fafc",
        card: "#1e293b",
        "card-foreground": "#f8fafc",
        popover: "#1e293b",
        "popover-foreground": "#f8fafc",
        primary: "#eab308",
        "primary-foreground": "#0f172a",
        secondary: "#334155",
        "secondary-foreground": "#f8fafc",
        muted: "#334155",
        "muted-foreground": "#94a3b8",
        accent: "#334155",
        "accent-foreground": "#f8fafc",
        destructive: "#ef4444",
        "destructive-foreground": "#f8fafc",
        border: "#334155",
        input: "#334155",
        ring: "#eab308",
      },
    },
  },
  plugins: [],
}
