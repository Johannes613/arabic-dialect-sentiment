import React from 'react';
import styled from 'styled-components';
import { Github, Twitter, Linkedin, BookOpen, ExternalLink } from 'lucide-react';

const FooterContainer = styled.footer`
  background: ${props => props.theme.colors.vercel.gray[950]};
  border-top: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing[12]} 0 ${props => props.theme.spacing[8]};
  margin-top: ${props => props.theme.spacing[16]};
`;

const FooterContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
`;

const FooterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing[8]};
  margin-bottom: ${props => props.theme.spacing[8]};
`;

const FooterSection = styled.div`
  h3 {
    font-size: ${props => props.theme.fontSizes.lg};
    margin-bottom: ${props => props.theme.spacing[4]};
    color: ${props => props.theme.colors.text.primary};
  }

  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  li {
    margin-bottom: ${props => props.theme.spacing[2]};
  }

  a {
    color: ${props => props.theme.colors.text.secondary};
    text-decoration: none;
    transition: color 0.2s ease;
    display: flex;
    align-items: center;
    gap: ${props => props.theme.spacing[2]};

    &:hover {
      color: ${props => props.theme.colors.primary};
    }
  }
`;

const FooterBottom = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-top: ${props => props.theme.spacing[6]};
  border-top: 1px solid ${props => props.theme.colors.border};

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    flex-direction: column;
    gap: ${props => props.theme.spacing[4]};
    text-align: center;
  }
`;

const Copyright = styled.p`
  color: ${props => props.theme.colors.text.muted};
  margin: 0;
`;

const SocialLinks = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing[4]};
`;

const SocialLink = styled.a`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: ${props => props.theme.borderRadius.full};
  background: ${props => props.theme.colors.vercel.gray[900]};
  color: ${props => props.theme.colors.text.secondary};
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.primary};
    color: ${props => props.theme.colors.text.inverse};
    transform: translateY(-2px);
  }
`;

const Footer = () => {
  return (
    <FooterContainer>
      <FooterContent>
        <FooterGrid>
          <FooterSection>
            <h3>Arabic Sentiment Analysis</h3>
            <p>
              Advanced sentiment analysis for Arabic dialects using domain-adapted 
              transformer models. Achieve 88% accuracy with our enhanced MARBERT architecture.
            </p>
          </FooterSection>

          <FooterSection>
            <h3>Quick Links</h3>
            <ul>
              <li><a href="/">Home</a></li>
              <li><a href="/docs">Documentation</a></li>
              <li><a href="/demo">Demo</a></li>
              <li>
                <a href="https://github.com/Johannes613/arabic-dialect-sentiment" target="_blank" rel="noopener noreferrer">
                  <Github size={16} />
                  Source Code
                </a>
              </li>
            </ul>
          </FooterSection>

          <FooterSection>
            <h3>Resources</h3>
            <ul>
              <li><a href="/docs#getting-started">Getting Started</a></li>
              <li><a href="/docs#api-reference">API Reference</a></li>
              <li><a href="/docs#examples">Examples</a></li>
              <li><a href="/docs#troubleshooting">Troubleshooting</a></li>
            </ul>
          </FooterSection>

          <FooterSection>
            <h3>Support & Community</h3>
            <ul>
              <li>
                <a href="https://github.com/Johannes613/arabic-dialect-sentiment/issues" target="_blank" rel="noopener noreferrer">
                  <ExternalLink size={16} />
                  Report Issues
                </a>
              </li>
              <li>
                <a href="https://github.com/Johannes613/arabic-dialect-sentiment/discussions" target="_blank" rel="noopener noreferrer">
                  <ExternalLink size={16} />
                  Discussions
                </a>
              </li>
              <li>
                <a href="https://github.com/Johannes613/arabic-dialect-sentiment/blob/main/README.md" target="_blank" rel="noopener noreferrer">
                  <BookOpen size={16} />
                  GitHub README
                </a>
              </li>
            </ul>
          </FooterSection>
        </FooterGrid>

        <FooterBottom>
          <Copyright>
            © 2025 Arabic Sentiment Analysis. A new era of sentiment analysis.
          </Copyright>
          <SocialLinks>
            <SocialLink href="https://github.com/Johannes613/arabic-dialect-sentiment" target="_blank" rel="noopener noreferrer">
              <Github size={20} />
            </SocialLink>
            <SocialLink href="https://x.com/john40336738581?s=21" target="_blank" rel="noopener noreferrer">
              <Twitter size={20} />
            </SocialLink>
            <SocialLink href="https://www.linkedin.com/in/yohannis-adamu-1837832b9?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" target="_blank" rel="noopener noreferrer">
              <Linkedin size={20} />
            </SocialLink>
          </SocialLinks>
        </FooterBottom>
      </FooterContent>
    </FooterContainer>
  );
};

export default Footer;
