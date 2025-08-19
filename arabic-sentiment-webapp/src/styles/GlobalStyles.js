import { createGlobalStyle } from 'styled-components';

const GlobalStyles = createGlobalStyle`
  * {
    box-sizing: border-box;
  }

  html {
    scroll-behavior: smooth;
  }

  body {
    margin: 0;
    padding: 0;
    font-family: ${props => props.theme.fonts.body};
    font-size: ${props => props.theme.fontSizes.base};
    line-height: 1.6;
    color: ${props => props.theme.colors.text.primary};
    background: ${props => props.theme.colors.background};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  h1, h2, h3, h4, h5, h6 {
    margin: 0 0 ${props => props.theme.spacing[4]} 0;
    font-weight: ${props => props.theme.fontWeights.bold};
    line-height: 1.2;
    color: ${props => props.theme.colors.text.primary};
  }

  p {
    margin: 0 0 ${props => props.theme.spacing[4]} 0;
    color: ${props => props.theme.colors.text.secondary};
  }

  a {
    color: ${props => props.theme.colors.accent};
    text-decoration: none;
    transition: color 0.2s ease;

    &:hover {
      color: ${props => props.theme.colors.primary};
    }
  }

  ul, ol {
    margin: 0 0 ${props => props.theme.spacing[4]} 0;
    padding-left: ${props => props.theme.spacing[6]};
    color: ${props => props.theme.colors.text.secondary};
  }

  li {
    margin-bottom: ${props => props.theme.spacing[2]};
  }

  button {
    font-family: inherit;
    cursor: pointer;
  }

  img {
    max-width: 100%;
    height: auto;
  }

  code {
    font-family: ${props => props.theme.fonts.mono};
    background: ${props => props.theme.colors.surface};
    padding: ${props => props.theme.spacing[1]} ${props => props.theme.spacing[2]};
    border-radius: ${props => props.theme.borderRadius.sm};
    font-size: ${props => props.theme.fontSizes.sm};
  }

  pre {
    font-family: ${props => props.theme.fonts.mono};
    background: ${props => props.theme.colors.surface};
    padding: ${props => props.theme.spacing[4]};
    border-radius: ${props => props.theme.borderRadius.md};
    overflow-x: auto;
    margin: ${props => props.theme.spacing[4]} 0;
  }

  blockquote {
    border-left: 4px solid ${props => props.theme.colors.border};
    margin: ${props => props.theme.spacing[4]} 0;
    padding-left: ${props => props.theme.spacing[4]};
    font-style: italic;
    color: ${props => props.theme.colors.text.secondary};
  }

  hr {
    border: none;
    border-top: 1px solid ${props => props.theme.colors.border};
    margin: ${props => props.theme.spacing[8]} 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: ${props => props.theme.spacing[4]} 0;
  }

  th, td {
    padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
    text-align: left;
    border-bottom: 1px solid ${props => props.theme.colors.border};
  }

  th {
    font-weight: ${props => props.theme.fontWeights.semibold};
    background: ${props => props.theme.colors.surface};
  }
`;

export default GlobalStyles;

