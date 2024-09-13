import { navbar } from "vuepress-theme-hope";

export const zhNavbar = navbar([
  "/zh/",
  {
    text: "软件开发",
    prefix: "/zh/software/",
    children: [
      { text: "PyQt5", icon: "sitemap", link: "pyqt5/" },
      { text: "前端开发", icon: "sitemap", link: "front_end/" },
      { text: "后端开发", icon: "sitemap", link: "back_end/" },
      { text: "桌面应用", icon: "sitemap", link: "desktop_app/" },
    ],
  },
  {
    text: "人工智能",
    prefix: "/zh/intelligence/",
    children: [
      { text: "机器学习", link: "MachineLearning/" },
      { text: "Numpy", link: "Numpy/" },
      { text: "Matplotlib", link: "Matplotlib/" },
      { text: "PyTorch", link: "PyTorch/" },
      { text: "推荐系统", link: "recommendationSystem/" },
    ],
  },
  {
    text: "玩出点名堂",
    prefix: "/zh/dobetter/",
    children: [
      { text: "麻将秘籍", prefix: "mahjong/", link: "mahjong/" },
      { text: "吉他乐理", prefix: "musictheroy/", link: "musictheroy/" },
      { text: "摄影摄像", prefix: "photograph/", link: "photograph/" },
    ],
  },
]);
