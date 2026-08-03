// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// TODO: replace with the real domain once registered (spec 00 §"Do these three things
// first", item 2). Sitemap and RSS both need this to be the canonical origin.
export default defineConfig({
  site: "https://patrickwrowe.github.io",
  trailingSlash: "always",
  integrations: [mdx(), sitemap()],
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
});
