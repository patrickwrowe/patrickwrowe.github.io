/// <reference types="astro/client" />

// @rollup/plugin-yaml turns a .yaml import into a plain JS object. The shape is not
// statically known, so /cv/ is the one place on the site without schema-checked data.
// If the CV grows, move it to a `file()` data collection with a Zod schema instead.
declare module "*.yaml" {
  const data: any;
  export default data;
}
