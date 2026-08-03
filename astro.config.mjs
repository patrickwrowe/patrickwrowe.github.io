// @ts-check
import { defineConfig, fontProviders } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import yaml from "@rollup/plugin-yaml";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// TODO: replace with the real domain once registered (spec 00 §"Do these three things
// first", item 2). Sitemap and RSS both need this to be the canonical origin.
export default defineConfig({
  site: "https://patrickwrowe.github.io",
  trailingSlash: "always",
  integrations: [mdx(), sitemap()],
  // Self-hosted via the Astro 6 Fonts API — spec 01 §5.2. Three roles, three faces:
  // sans labels, serif reads, mono is data.
  fonts: [
    {
      provider: fontProviders.google(),
      name: "Instrument Sans",
      cssVariable: "--font-sans",
      weights: [400, 500, 600],
      subsets: ["latin"],
      fallbacks: ["system-ui", "sans-serif"],
    },
    {
      provider: fontProviders.google(),
      name: "Source Serif 4",
      cssVariable: "--font-serif",
      weights: [400, 600],
      subsets: ["latin"],
      fallbacks: ["Georgia", "serif"],
    },
    {
      provider: fontProviders.google(),
      name: "IBM Plex Mono",
      cssVariable: "--font-mono",
      weights: [400, 500],
      subsets: ["latin"],
      fallbacks: ["ui-monospace", "monospace"],
    },
  ],
  markdown: {
    // DEVIATION from spec 02 §3.2, which puts remarkPlugins/rehypePlugins directly on
    // `markdown`. Astro 6.4 deprecated that shape in favour of `markdown.processor`
    // with `unified()`. Same plugins, same build-time KaTeX rendering — only the
    // wiring differs. Update the spec snippet when convenient.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
    shikiConfig: { theme: "min-light", wrap: true },
  },
  // Lets /cv/ import src/data/cv.yaml directly, keeping the YAML the single source
  // spec 02 §5 calls for. The alternative — generating a JSON twin — would mean two
  // files to keep in sync, which is the thing that section exists to prevent.
  vite: { plugins: [yaml()] },
});
