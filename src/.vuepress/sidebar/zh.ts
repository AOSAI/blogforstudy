import { sidebar } from "vuepress-theme-hope";

export const zhSidebar = sidebar({
  // 人工智能
  "/zh/intelligence/MachineLearning/": "structure",
  "/zh/intelligence/Numpy/": "structure",
  "/zh/intelligence/Matplotlib/": "structure",
  "/zh/intelligence/recommendationSystem/": [
    {
      text: "推荐系统",
      icon: "tablet-alt",
      prefix: "",
      children: "structure",
    },
  ],
  // 软件开发
  "/zh/software/pyqt5/": "structure",
  "/zh/software/": [
    {
      text: "前端开发",
      icon: "tablet-alt",
      prefix: "front_end/",
      // link: "codes/software/front_end/",
      children: "structure",
    },
    {
      text: "后端开发",
      icon: "sitemap",
      prefix: "back_end/",
      children: "structure",
    },
    {
      text: "桌面应用",
      icon: "laptop-code",
      prefix: "desktop_app/",
      children: "structure",
    },
  ],
  // 班门弄斧
  "/zh/dobetter/mahjong/": "structure",
  "/zh/dobetter/musictheroy/": "structure",
  "/zh/dobetter/photograph/": "structure",
});
