/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#070B14',
        'bg-soft': '#0C1220',
        surface: '#172033',
        'surface-lavender': '#241D46',
        primary: '#6958F5',
        'primary-hover': '#7C6BFF',
        'primary-active': '#5544E8',
        'primary-light': '#9488FF',
        lavender: '#7C6BFF',
        'lavender-soft': '#2B2450',
        'text-main': '#F7F8FC',
        'text-secondary': '#C2C8D6',
        'text-muted': '#929BAD',
        border: '#2C374B',
        'border-strong': '#43506A',
        success: '#54D68C',
        warning: '#F6B94A',
        error: '#E75A68',
        info: '#3B82F6',
      },
      fontFamily: {
        sans: ['Poppins', 'Shabnam', 'Tahoma', 'Arial', 'sans-serif'],
        serif: ['Playfair Display', 'Shabnam', 'serif'],
        fa: ['Shabnam', 'Tahoma', 'Arial', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
