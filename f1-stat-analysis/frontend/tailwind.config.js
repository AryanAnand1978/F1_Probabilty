/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        f1red: "#e10600",
        f1black: "#111111",
        f1gray: "#1f1f1f",
      },
      boxShadow: {
        card: "0 10px 30px rgba(0, 0, 0, 0.25)",
      },
    },
  },
  plugins: [],
};
