/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{html,ts}"
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                cyberBlue: '#2563eb',
                cyberViolet: '#7c3aed'
            }
        }
    },
    plugins: []
}
