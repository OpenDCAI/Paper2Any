/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['"Inter"', '"Helvetica Neue"', 'Arial', 'sans-serif'],
      },
      colors: {
        primary: {
          50: '#faf4f5',
          100: '#f4e7ea',
          200: '#e6cfd6',
          300: '#d4aeb8',
          400: '#b97989',
          500: '#8f3147',
          600: '#7d2b40',
          700: '#672537',
          800: '#4e1d2c',
          900: '#2f121b',
        },
        glass: {
          light: 'rgba(255, 252, 247, 0.72)',
          medium: 'rgba(255, 249, 242, 0.56)',
          dark: 'rgba(62, 23, 37, 0.82)',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 10px 30px rgba(143, 49, 71, 0.12)' },
          '100%': { boxShadow: '0 18px 48px rgba(143, 49, 71, 0.22)' },
        }
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        shell: '0 24px 80px rgba(117, 36, 57, 0.14)',
        panel: '0 20px 50px rgba(92, 36, 52, 0.12)',
      },
    },
  },
  plugins: [],
}
