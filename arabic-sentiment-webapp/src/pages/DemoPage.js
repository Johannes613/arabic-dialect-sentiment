import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Send, RotateCcw, Zap, Target, BarChart3 } from 'lucide-react';

const PageContainer = styled.div`
  padding-top: 64px;
  min-height: calc(100vh - 64px);
  background: linear-gradient(135deg, ${props => props.theme.colors.background} 0%, ${props => props.theme.colors.surface} 100%);
`;

const DemoHeader = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing[16]} 0 ${props => props.theme.spacing[8]};
  max-width: 800px;
  margin: 0 auto;
`;

const DemoTitle = styled.h1`
  font-size: ${props => props.theme.fontSizes['4xl']};
  margin-bottom: ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.text.primary};
`;

const DemoSubtitle = styled.p`
  font-size: ${props => props.theme.fontSizes.lg};
  color: ${props => props.theme.colors.text.secondary};
  margin-bottom: ${props => props.theme.spacing[6]};
  line-height: 1.6;
`;

const DemoContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[6]} ${props => props.theme.spacing[12]};
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing[8]};

  @media (max-width: ${props => props.theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
    gap: ${props => props.theme.spacing[6]};
  }
`;

const InputSection = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing[6]};
  box-shadow: ${props => props.theme.shadows.md};
`;

const SectionTitle = styled.h2`
  font-size: ${props => props.theme.fontSizes.xl};
  margin-bottom: ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.text.primary};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 120px;
  padding: ${props => props.theme.spacing[4]};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  font-family: inherit;
  font-size: ${props => props.theme.fontSizes.base};
  resize: vertical;
  transition: border-color 0.2s ease;
  background: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.text.primary};

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.accent};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.accent}20;
  }

  &::placeholder {
    color: ${props => props.theme.colors.text.muted};
  }
`;

const Button = styled.button`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[6]};
  background: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.text.inverse};
  border: none;
  border-radius: ${props => props.theme.borderRadius.md};
  font-weight: ${props => props.theme.fontWeights.medium};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: ${props => props.theme.fontSizes.base};

  &:hover:not(:disabled) {
    background: ${props => props.theme.colors.vercel.gray[200]};
    transform: translateY(-1px);
    box-shadow: ${props => props.theme.shadows.lg};
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SecondaryButton = styled(Button)`
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text.primary};
  border: 1px solid ${props => props.theme.colors.border};

  &:hover:not(:disabled) {
    background: ${props => props.theme.colors.surface};
    border-color: ${props => props.theme.colors.primary};
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing[3]};
  margin-top: ${props => props.theme.spacing[4]};
  flex-wrap: wrap;
`;

const ResultsSection = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  padding: ${props => props.theme.spacing[6]};
  box-shadow: ${props => props.theme.shadows.md};
`;

const ResultCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing[4]};
  margin-bottom: ${props => props.theme.spacing[4]};
  transition: all 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const ResultHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${props => props.theme.spacing[3]};
`;

const SentimentLabel = styled.span`
  padding: ${props => props.theme.spacing[1]} ${props => props.theme.spacing[3]};
  border-radius: ${props => props.theme.borderRadius.full};
  font-weight: ${props => props.theme.fontWeights.medium};
  font-size: ${props => props.theme.fontSizes.sm};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const SentimentPOS = styled(SentimentLabel)`
  background: #10b981;
  color: white;
`;

const SentimentNEG = styled(SentimentLabel)`
  background: #ef4444;
  color: white;
`;

const SentimentNEUTRAL = styled(SentimentLabel)`
  background: #6b7280;
  color: white;
`;

const SentimentOBJ = styled(SentimentLabel)`
  background: #4b5563;
  color: white;
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 8px;
  background: ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.full};
  overflow: hidden;
  margin: ${props => props.theme.spacing[2]} 0;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: ${props => props.theme.colors.primary};
  border-radius: ${props => props.theme.borderRadius.full};
  transition: width 0.3s ease;
  width: ${props => props => props.confidence * 100}%;
`;

const ConfidenceText = styled.div`
  font-size: ${props => props.theme.fontSizes.sm};
  color: ${props => props.theme.colors.text.muted};
  text-align: right;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${props => props.theme.spacing[4]};
  margin-top: ${props => props.theme.spacing[6]};
`;

const StatCard = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing[4]};
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius.md};
  border: 1px solid ${props => props.theme.colors.border};
`;

const StatNumber = styled.div`
  font-size: ${props => props.theme.fontSizes['2xl']};
  font-weight: ${props => props.theme.fontWeights.bold};
  color: ${props => props.theme.colors.accent};
  margin-bottom: ${props => props.theme.spacing[1]};
`;

const StatLabel = styled.div`
  font-size: ${props => props.theme.fontSizes.sm};
  color: ${props => props.theme.colors.text.secondary};
`;

const SampleTexts = styled.div`
  margin-top: ${props => props.theme.spacing[4]};
`;

const SampleText = styled.button`
  display: block;
  width: 100%;
  text-align: left;
  padding: ${props => props.theme.spacing[2]} ${props => props.theme.spacing[3]};
  background: none;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  margin-bottom: ${props => props.theme.spacing[2]};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: ${props => props.theme.fontSizes.sm};
  color: ${props => props.theme.colors.text.secondary};

  &:hover {
    background: ${props => props.theme.colors.surface};
    border-color: ${props => props.theme.colors.primary};
    color: ${props => props.theme.colors.text.primary};
  }
`;

const DemoPage = () => {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stats, setStats] = useState({
    totalAnalyzed: 0,
    averageConfidence: 0,
    processingTime: 0
  });

  const sampleTexts = [
    "هذا المنتج ممتاز جدا وأوصي به بشدة",
    "الخدمة سيئة للغاية ولا أنصح بها",
    "التجربة كانت عادية، لا أكثر ولا أقل",
    "أنا محايد في هذا الموضوع ولا أملك رأي واضح",
    "المطعم رائع والطعام لذيذ",
    "الفيلم ممل جدا وضياع للوقت",
    "الكتاب مفيد ويحتوي على معلومات قيمة",
    "الطقس جميل اليوم والجو منعش"
  ];

  const getSentimentLabel = (sentiment) => {
    switch (sentiment) {
      case 'POS': return SentimentPOS;
      case 'NEG': return SentimentNEG;
      case 'NEUTRAL': return SentimentNEUTRAL;
      case 'OBJ': return SentimentOBJ;
      default: return SentimentLabel;
    }
  };

  const mockAnalyzeSentiment = async (text) => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
    
    const sentiments = ['POS', 'NEG', 'NEUTRAL', 'OBJ'];
    const sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    const confidence = 0.7 + Math.random() * 0.3; // 0.7 to 1.0
    const processingTime = 0.1 + Math.random() * 0.2; // 0.1 to 0.3 seconds
    
    return {
      text,
      sentiment,
      confidence,
      processing_time: processingTime,
      dialect: 'gulf'
    };
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    
    setIsAnalyzing(true);
    const result = await mockAnalyzeSentiment(inputText);
    
    setResults(prev => [result, ...prev]);
    setInputText('');
    
    // Update stats
    setStats(prev => ({
      totalAnalyzed: prev.totalAnalyzed + 1,
      averageConfidence: ((prev.averageConfidence * prev.totalAnalyzed) + result.confidence) / (prev.totalAnalyzed + 1),
      processingTime: result.processing_time
    }));
    
    setIsAnalyzing(false);
  };

  const handleSampleText = (text) => {
    setInputText(text);
  };

  const handleReset = () => {
    setResults([]);
    setStats({
      totalAnalyzed: 0,
      averageConfidence: 0,
      processingTime: 0
    });
  };

  const handleBatchAnalyze = async () => {
    if (sampleTexts.length === 0) return;
    
    setIsAnalyzing(true);
    const batchResults = [];
    
    for (const text of sampleTexts) {
      const result = await mockAnalyzeSentiment(text);
      batchResults.push(result);
    }
    
    setResults(batchResults);
    
    // Update stats
    const avgConfidence = batchResults.reduce((sum, r) => sum + r.confidence, 0) / batchResults.length;
    const avgProcessingTime = batchResults.reduce((sum, r) => sum + r.processing_time, 0) / batchResults.length;
    
    setStats({
      totalAnalyzed: batchResults.length,
      averageConfidence: avgConfidence,
      processingTime: avgProcessingTime
    });
    
    setIsAnalyzing(false);
  };

  return (
    <PageContainer>
      <DemoHeader>
        <DemoTitle>Arabic Sentiment Analysis Demo</DemoTitle>
        <DemoSubtitle>
          Experience our enhanced MARBERT model in action. Enter Arabic text to analyze sentiment 
          or try our sample texts to see the system's capabilities.
        </DemoSubtitle>
      </DemoHeader>

      <DemoContainer>
        <InputSection>
          <SectionTitle>
            <Target size={20} />
            Input Text
          </SectionTitle>
          
          <TextArea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="أدخل النص العربي هنا للتحليل... (Enter Arabic text here for analysis...)"
            disabled={isAnalyzing}
          />
          
          <ButtonGroup>
            <Button onClick={handleAnalyze} disabled={isAnalyzing || !inputText.trim()}>
              {isAnalyzing ? (
                <>
                  <RotateCcw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                  Analyzing...
                </>
              ) : (
                <>
                  <Send size={16} />
                  Analyze Sentiment
                </>
              )}
            </Button>
            
            <SecondaryButton onClick={handleBatchAnalyze} disabled={isAnalyzing}>
              <Zap size={16} />
              Analyze Samples
            </SecondaryButton>
            
            <SecondaryButton onClick={handleReset} disabled={isAnalyzing}>
              <RotateCcw size={16} />
              Reset
            </SecondaryButton>
          </ButtonGroup>

          <SampleTexts>
            <h4 style={{ marginBottom: '1rem', color: '#666' }}>Sample Texts:</h4>
            {sampleTexts.map((text, index) => (
              <SampleText key={index} onClick={() => handleSampleText(text)}>
                {text}
              </SampleText>
            ))}
          </SampleTexts>
        </InputSection>

        <ResultsSection>
          <SectionTitle>
            <BarChart3 size={20} />
            Analysis Results
          </SectionTitle>
          
          {results.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
              <BarChart3 size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
              <p>No analysis results yet. Enter some text and click "Analyze Sentiment" to get started.</p>
            </div>
          ) : (
            <>
              {results.map((result, index) => {
                const SentimentComponent = getSentimentLabel(result.sentiment);
                return (
                  <ResultCard
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <ResultHeader>
                      <SentimentComponent>{result.sentiment}</SentimentComponent>
                      <span style={{ fontSize: '0.875rem', color: '#666' }}>
                        {result.processing_time.toFixed(3)}s
                      </span>
                    </ResultHeader>
                    
                    <p style={{ marginBottom: '0.5rem', lineHeight: 1.5 }}>{result.text}</p>
                    
                    <ConfidenceBar>
                      <ConfidenceFill confidence={result.confidence} />
                    </ConfidenceBar>
                    <ConfidenceText>Confidence: {(result.confidence * 100).toFixed(1)}%</ConfidenceText>
                  </ResultCard>
                );
              })}

              <StatsGrid>
                <StatCard>
                  <StatNumber>{stats.totalAnalyzed}</StatNumber>
                  <StatLabel>Total Analyzed</StatLabel>
                </StatCard>
                <StatCard>
                  <StatNumber>{(stats.averageConfidence * 100).toFixed(1)}%</StatNumber>
                  <StatLabel>Avg Confidence</StatLabel>
                </StatCard>
                <StatCard>
                  <StatNumber>{(stats.processingTime * 1000).toFixed(0)}ms</StatNumber>
                  <StatLabel>Avg Processing</StatLabel>
                </StatCard>
              </StatsGrid>
            </>
          )}
        </ResultsSection>
      </DemoContainer>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </PageContainer>
  );
};

export default DemoPage;

