import type { DocsLayoutProps } from 'fumadocs-ui/layout';

export const baseOptions: Partial<DocsLayoutProps> = {
  nav: {
    title: 'lux-crypto',
  },
  links: [
    {
      text: 'Documentation',
      url: '/docs',
      active: 'nested-url',
    },
    {
      text: 'GitHub',
      url: 'https://github.com/luxfi/crypto',
    },
  ],
};
