import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Course',
  tagline: 'An educational course book on Physical AI & Humanoid Robotics',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://your-physical-ai-course-site.example.com',
  baseUrl: '/',

  organizationName: 'physical-ai',
  projectName: 'physical-ai',

  onBrokenLinks: 'throw',

  // ❗ FIXED (deprecated): moved from root → markdown.hooks
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  // clientModules: [
  //   require.resolve('./src/theme/ProviderWrapper.tsx'),
  // ],

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },

        // blog: {
        //   showReadingTime: true,
        //   feedOptions: {
        //     type: ['rss', 'atom'],
        //     xslt: true,
        //   },
        //   editUrl:
        //     'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',

        //   // keeps helpful warnings
        //   onInlineTags: 'warn',
        //   onInlineAuthors: 'warn',
        //   onUntruncatedBlogPosts: 'warn',
        // },

        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/physical-ai-social-card.jpg',

    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Physical AI Book',
      logo: {
        alt: 'Physical AI ',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Course Book',
        },
        // { to: '/blog', label: 'Blog', position: 'left' },
        // {
        //   type: 'custom-authButtons',
        //   position: 'right',
        // },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/physical-ai/physical-robotics-course',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course Modules',
          items: [
            { label: 'Module 1: Robotic Nervous System', to: '/docs/module-1-ros2-nervous-system/overview' },
            { label: 'Module 2: Digital Twin Simulation', to: '/docs/module-2-digital-twin-simulation/overview' },
            { label: 'Module 3: NVIDIA Isaac AI Brain', to: '/docs/module-3-nvidia-isaac-ai-brain/overview' },
            { label: 'Module 4: Vision-Language-Action', to: '/docs/module-4-vision-language-action/overview' },
          ],
        },
        // {
        //   title: 'Essential Resources',
        //   items: [
        //     // { label: 'Course Introduction', to: '/docs/intro' },
        //     { label: 'Learning Outcomes', to: '/docs/learning-outcomes' },
           
        //     { label: 'Weekly Breakdown', to: '/docs/weekly-breakdown' },
        //     { label: 'why-physical-ai-matters', to: '/docs/why-physical-ai-matters' },
        //   ],
        // },
        {
          title: 'Community',
          items: [
            { label: 'Discord', href: 'https://discordapp.com/invite/physical-ai' },
            { label: 'X', href: 'https://x.com/physical_ai' },
            { label: 'GitHub', href: 'https://github.com/physical-ai/physical-ai-robotics-course' },
            { label: 'Blog', to: '/blog' },
          ],
        },

      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Course. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

// import {themes as prismThemes} from 'prism-react-renderer';
// import type {Config} from '@docusaurus/types';
// import type * as Preset from '@docusaurus/preset-classic';

// // This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// const config: Config = {
//   title: 'Physical AI & Humanoid Robotics Course',
//   tagline: 'An educational course book on Physical AI & Humanoid Robotics',
//   favicon: 'img/favicon.ico',

//   // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
//   future: {
//     v4: true, // Improve compatibility with the upcoming Docusaurus v4
//   },

//   // Set the production url of your site here
//   url: 'https://your-physical-ai-course-site.example.com',
//   // Set the /<baseUrl>/ pathname under which your site is served
//   // For GitHub pages deployment, it is often '/<projectName>/'
//   baseUrl: '/',

//   // GitHub pages deployment config.
//   // If you aren't using GitHub pages, you don't need these.
//   organizationName: 'physical-ai', // Usually your GitHub org/user name.
//   projectName: 'physical-ai-robotics-course', // Usually your repo name.

//   onBrokenLinks: 'throw',
//   onBrokenMarkdownLinks: 'warn',

//   // Even if you don't use internationalization, you can use this field to set
//   // useful metadata like html lang. For example, if your site is Chinese, you
//   // may want to replace "en" with "zh-Hans".
//   i18n: {
//     defaultLocale: 'en',
//     locales: ['en', 'ur'], // Adding Urdu locale as per requirements
//   },

//   presets: [
//     [
//       'classic',
//       {
//         docs: {
//           sidebarPath: './sidebars.ts',
//           // Please change this to your repo.
//           // Remove this to remove the "edit this page" links.
//           editUrl:
//             'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
//         },
//         blog: {
//           showReadingTime: true,
//           feedOptions: {
//             type: ['rss', 'atom'],
//             xslt: true,
//           },
//           // Please change this to your repo.
//           // Remove this to remove the "edit this page" links.
//           editUrl:
//             'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
//           // Useful options to enforce blogging best practices
//           onInlineTags: 'warn',
//           onInlineAuthors: 'warn',
//           onUntruncatedBlogPosts: 'warn',
//         },
//         theme: {
//           customCss: './src/css/custom.css',
//         },
//       } satisfies Preset.Options,
//     ],
//   ],

//   themeConfig: {
//     // Replace with your project's social card
//     image: 'img/physical-ai-social-card.jpg',
//     colorMode: {
//       respectPrefersColorScheme: true,
//     },
//     navbar: {
//       title: 'Physical AI & Humanoid Robotics',
//       logo: {
//         alt: 'Physical AI & Humanoid Robotics Logo',
//         src: 'img/logo.svg',
//       },
//       items: [
//         {
//           type: 'docSidebar',
//           sidebarId: 'tutorialSidebar',
//           position: 'left',
//           label: 'Course Book',
//         },
//         {to: '/blog', label: 'Blog', position: 'left'},
//         // Adding Urdu translation option at the top of navbar as required
//         {
//           type: 'localeDropdown',
//           position: 'right',
//         },
//         {
//           href: 'https://github.com/physical-ai/physical-ai-robotics-course',
//           label: 'GitHub',
//           position: 'right',
//         },
//       ],
//     },
//     footer: {
//       style: 'dark',
//       links: [
//         {
//           title: 'Course Content',
//           items: [
//             {
//               label: 'Introduction',
//               to: '/docs/intro',
//             },
//             {
//               label: 'Module 1: The Robotic Nervous System',
//               to: '/docs/module-1-ros2-nervous-system/overview',
//             },
//           ],
//         },
//         {
//           title: 'Community',
//           items: [
//             {
//               label: 'Discord',
//               href: 'https://discordapp.com/invite/physical-ai',
//             },
//             {
//               label: 'X',
//               href: 'https://x.com/physical_ai',
//             },
//           ],
//         },
//         {
//           title: 'More',
//           items: [
//             {
//               label: 'Blog',
//               to: '/blog',
//             },
//             {
//               label: 'GitHub',
//               href: 'https://github.com/physical-ai/physical-ai-robotics-course',
//             },
//           ],
//         },
//       ],
//       copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Course. Built with Docusaurus.`,
//     },
//     prism: {
//       theme: prismThemes.github,
//       darkTheme: prismThemes.dracula,
//     },
//   } satisfies Preset.ThemeConfig,
// };

// export default config;
