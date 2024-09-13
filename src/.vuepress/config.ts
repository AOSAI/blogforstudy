import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/blog-pages/",

  locales: {
    "/": {
      lang: "en-US",
      title: "Blog by AOSAI",
      description: "Blog by AOSAI which used vuepress-theme-hope theme",
    },
    "/zh/": {
      lang: "zh-CN",
      title: "青裁的博客",
      description: "使用了vuepress-theme-hope主题",
    },
  },

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
