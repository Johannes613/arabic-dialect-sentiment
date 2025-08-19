import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowRight, Zap, Target, BarChart3, Code, BookOpen, Play, Github } from 'lucide-react';

const PageContainer = styled.div`
  padding-top: 64px;
`;

const HeroSection = styled.section`
  background: linear-gradient(135deg, ${props => props.theme.colors.background} 0%, ${props => props.theme.colors.vercel.gray[950]} 100%);
  padding: ${props => props.theme.spacing[24]} 0 ${props => props.theme.spacing[16]};
  text-align: center;
  min-height: 80vh;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const HeroContent = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
`;

const HeroTitle = styled(motion.h1)`
  font-size: ${props => props.theme.fontSizes['5xl']};
  font-weight: ${props => props.theme.fontWeights.bold};
  margin-bottom: ${props => props.theme.spacing[6]};
  line-height: 1.1;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.primary} 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const HeroSubtitle = styled(motion.p)`
  font-size: ${props => props.theme.fontSizes.xl};
  color: ${props => props.theme.colors.text.secondary};
  margin-bottom: ${props => props.theme.spacing[8]};
  line-height: 1.6;
`;

const HeroButtons = styled(motion.div)`
  display: flex;
  gap: ${props => props.theme.spacing[4]};
  justify-content: center;
  flex-wrap: wrap;
`;

const Button = styled(Link)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
  padding: ${props => props.theme.spacing[4]} ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.lg};
  font-weight: ${props => props.theme.fontWeights.medium};
  text-decoration: none;
  transition: all 0.2s ease;
  font-size: ${props => props.theme.fontSizes.lg};
`;

const PrimaryButton = styled(Button)`
  background: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.text.inverse};

  &:hover {
    background: ${props => props.theme.colors.vercel.gray[200]};
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.lg};
  }
`;

const SecondaryButton = styled(Button)`
  background: ${props => props.theme.colors.vercel.gray[900]};
  color: ${props => props.theme.colors.text.primary};
  border: 2px solid ${props => props.theme.colors.border};

  &:hover {
    border-color: ${props => props.theme.colors.primary};
    color: ${props => props.theme.colors.primary};
    transform: translateY(-2px);
  }
`;

const StatsSection = styled.section`
  padding: ${props => props.theme.spacing[16]} 0;
  background: ${props => props.theme.colors.background};
`;

const StatsGrid = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing[8]};
`;

const StatCard = styled(motion.div)`
  text-align: center;
  padding: ${props => props.theme.spacing[6]};
  background: ${props => props.theme.colors.vercel.gray[900]};
  border-radius: ${props => props.theme.borderRadius.lg};
  border: 1px solid ${props => props.theme.colors.border};
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.lg};
  }
`;

const StatNumber = styled.div`
  font-size: ${props => props.theme.fontSizes['4xl']};
  font-weight: ${props => props.theme.fontWeights.bold};
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing[2]};
`;

const StatLabel = styled.div`
  color: ${props => props.theme.colors.text.secondary};
  font-weight: ${props => props.theme.fontWeights.medium};
`;

const FeaturesSection = styled.section`
  padding: ${props => props.theme.spacing[16]} 0;
  background: ${props => props.theme.colors.vercel.gray[950]};
`;

const SectionTitle = styled.h2`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing[12]};
  font-size: ${props => props.theme.fontSizes['4xl']};
`;

const FeaturesGrid = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing[8]};
`;

const FeatureCard = styled(motion.div)`
  background: ${props => props.theme.colors.vercel.gray[900]};
  padding: ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.lg};
  border: 1px solid ${props => props.theme.colors.border};
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.lg};
  }
`;

const FeatureIcon = styled.div`
  width: 60px;
  height: 60px;
  background: ${props => props.theme.colors.primary};
  border-radius: ${props => props.theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.text.inverse};
`;

const FeatureTitle = styled.h3`
  font-size: ${props => props.theme.fontSizes.xl};
  margin-bottom: ${props => props.theme.spacing[3]};
  color: ${props => props.theme.colors.text.primary};
`;

const FeatureDescription = styled.p`
  color: ${props => props.theme.colors.text.secondary};
  line-height: 1.6;
`;

const CTASection = styled.section`
  padding: ${props => props.theme.spacing[16]} 0;
  background: ${props => props.theme.colors.background};
  text-align: center;
`;

const CTAContent = styled.div`
  max-width: 600px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
`;

const LandingPage = () => {
  const stats = [
    { number: '88%', label: 'Accuracy' },
    { number: '86%', label: 'Macro F1' },
    { number: '4', label: 'Sentiment Classes' },
    { number: '500', label: 'Samples per Class' }
  ];

  const features = [
    {
      icon: <Zap size={24} />,
      title: 'Enhanced MARBERT',
      description: 'Domain-adapted transformer model specifically optimized for Arabic dialect sentiment analysis with state-of-the-art performance.'
    },
    {
      icon: <Target size={24} />,
      title: 'Class Balancing',
      description: 'Advanced techniques including undersampling, oversampling, and data augmentation to handle class imbalance effectively.'
    },
    {
      icon: <BarChart3 size={24} />,
      title: 'Performance Metrics',
      description: 'Comprehensive evaluation with Macro F1, precision, recall, and confusion matrix analysis for all sentiment classes.'
    },
    {
      icon: <Code size={24} />,
      title: 'Production Ready',
      description: 'Optimized for deployment with FastAPI backend, React frontend, and Docker containerization.'
    },
    {
      icon: <BookOpen size={24} />,
      title: 'Comprehensive Docs',
      description: 'Detailed documentation, examples, and API reference for easy integration and customization.'
    },
    {
      icon: <Play size={24} />,
      title: 'Live Demo',
      description: 'Interactive demonstration showcasing the model capabilities with real-time sentiment analysis.'
    }
  ];

  return (
    <PageContainer>
      <HeroSection>
        <HeroContent>
          <HeroTitle
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Arabic Dialect Sentiment Analysis
          </HeroTitle>
          <HeroSubtitle
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            Advanced sentiment analysis for Arabic dialects using domain-adapted transformer models. 
            Achieve 88% accuracy with our enhanced MARBERT architecture.
          </HeroSubtitle>
          <HeroButtons
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <PrimaryButton to="/demo">
              Try Demo
              <ArrowRight size={20} />
            </PrimaryButton>
            <SecondaryButton to="/docs">
              View Documentation
              <BookOpen size={20} />
            </SecondaryButton>
            <SecondaryButton as="a" href="https://github.com/Johannes613/arabic-dialect-sentiment" target="_blank" rel="noopener noreferrer">
              <Github size={20} />
              Source Code
            </SecondaryButton>
          </HeroButtons>
        </HeroContent>
      </HeroSection>

      <StatsSection>
        <StatsGrid>
          {stats.map((stat, index) => (
            <StatCard
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <StatNumber>{stat.number}</StatNumber>
              <StatLabel>{stat.label}</StatLabel>
            </StatCard>
          ))}
        </StatsGrid>
      </StatsSection>

      <FeaturesSection>
        <SectionTitle>Key Features</SectionTitle>
        <FeaturesGrid>
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <FeatureIcon>{feature.icon}</FeatureIcon>
              <FeatureTitle>{feature.title}</FeatureTitle>
              <FeatureDescription>{feature.description}</FeatureDescription>
            </FeatureCard>
          ))}
        </FeaturesGrid>
      </FeaturesSection>

      <CTASection>
        <CTAContent>
          <h2>Ready to Get Started?</h2>
          <p style={{ marginBottom: '2rem', fontSize: '1.125rem', color: '#a1a1aa' }}>
            Explore our documentation, try the live demo, or integrate our API into your applications.
          </p>
          <HeroButtons>
            <PrimaryButton to="/demo">
              Start with Demo
              <Play size={20} />
            </PrimaryButton>
            <SecondaryButton to="/docs">
              Read Documentation
              <BookOpen size={20} />
            </SecondaryButton>
          </HeroButtons>
        </CTAContent>
      </CTASection>
    </PageContainer>
  );
};

export default LandingPage;

