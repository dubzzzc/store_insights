/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/static/**/*.html",
    "./app/static/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        // Design System Colors
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
          50: "hsl(221, 83%, 95%)",
          100: "hsl(221, 83%, 90%)",
          200: "hsl(221, 83%, 80%)",
          300: "hsl(221, 83%, 70%)",
          400: "hsl(221, 83%, 60%)",
          500: "hsl(221, 83%, 53%)",
          600: "hsl(221, 83%, 45%)",
          700: "hsl(221, 83%, 35%)",
          800: "hsl(221, 83%, 25%)",
          900: "hsl(221, 83%, 15%)",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      boxShadow: {
        'shadow-1': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'shadow-2': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'shadow-3': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
      },
      spacing: {
        // 8pt grid system
        '8pt': '8px',
        '16pt': '16px',
        '24pt': '24px',
        '32pt': '32px',
        '40pt': '40px',
        '48pt': '48px',
        '56pt': '56px',
        '64pt': '64px',
      },
    },
  },
  plugins: [],
}

