// // @ts-check
// import { defineConfig } from "astro/config";
// import tailwindcss from "@tailwindcss/vite";
// // https://astro.build/config
// export default defineConfig({
//   vite: {
//     plugins: [tailwindcss()],
//   }
// });

import { defineConfig } from 'astro/config';
import tailwindcss from "@tailwindcss/vite";


// https://astro.build/config
export default defineConfig({
  site: 'https://dreys-bot.github.io/DPBAPortfolio/', // Mets bien l'URL ici
  trailingSlash: 'always',
  vite: {
  plugins: [tailwindcss()],
  }
});
