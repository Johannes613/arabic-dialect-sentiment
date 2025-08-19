const theme = {
  colors: {
    primary: '#ffffff',
    secondary: '#a1a1aa',
    accent: '#ffffff',
    background: '#000000',
    surface: '#111111',
    border: '#333333',
    text: {
      primary: '#ffffff',
      secondary: '#a1a1aa',
      muted: '#71717a',
      inverse: '#000000'
    },
    success: '#10b981',
    warning: '#f5a623',
    error: '#e00',
    // Vercel-specific colors
    vercel: {
      black: '#000000',
      white: '#ffffff',
      gray: {
        50: '#fafafa',
        100: '#f5f5f5',
        200: '#e5e5e5',
        300: '#d4d4d4',
        400: '#a3a3a3',
        500: '#737373',
        600: '#525252',
        700: '#404040',
        800: '#262626',
        900: '#171717',
        950: '#0a0a0a'
      },
      blue: {
        50: '#ffffff',
        100: '#ffffff',
        200: '#ffffff',
        300: '#ffffff',
        400: '#ffffff',
        500: '#ffffff',
        600: '#ffffff',
        700: '#ffffff',
        800: '#ffffff',
        900: '#ffffff',
        950: '#ffffff'
      }
    }
  },
  fonts: {
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
    mono: 'SF Mono, Monaco, Inconsolata, "Roboto Mono", "Bitstream Vera Sans Mono", "Courier New", monospace'
  },
  fontSizes: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
    '4xl': '2.25rem',
    '5xl': '3rem',
    '6xl': '3.75rem'
  },
  fontWeights: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700
  },
  spacing: {
    0: '0',
    1: '0.25rem',
    2: '0.5rem',
    3: '0.75rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    8: '2rem',
    10: '2.5rem',
    12: '3rem',
    16: '4rem',
    20: '5rem',
    24: '6rem',
    32: '8rem'
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px'
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3)'
  },
  borderRadius: {
    sm: '4px',
    md: '6px',
    lg: '8px',
    xl: '12px',
    full: '9999px'
  }
};

export default theme;

