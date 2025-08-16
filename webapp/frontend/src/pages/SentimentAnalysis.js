import React, { useState } from 'react';
import { useMutation } from 'react-query';
import { toast } from 'react-hot-toast';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Languages, 
  Clock, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  XCircle,
  MinusCircle
} from 'lucide-react';

// API service
import { analyzeSentiment } from '../services/api';

// Components
import LoadingSpinner from '../components/LoadingSpinner';
import SentimentChart from '../components/SentimentChart';
import ExplanationPanel from '../components/ExplanationPanel';

const SentimentAnalysis = () => {
  const [text, setText] = useState('');
  const [dialect, setDialect] = useState('');
  const [includeExplanation, setIncludeExplanation] = useState(false);
  const [results, setResults] = useState(null);

  // Sentiment analysis mutation
  const sentimentMutation = useMutation(analyzeSentiment, {
    onSuccess: (data) => {
      setResults(data);
      toast.success('تم تحليل النص بنجاح!');
    },
    onError: (error) => {
      toast.error(`فشل في تحليل النص: ${error.message}`);
    },
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      toast.error('يرجى إدخال نص للتحليل');
      return;
    }

    sentimentMutation.mutate({
      text: text.trim(),
      dialect: dialect || undefined,
      include_explanation: includeExplanation,
    });
  };

  const handleClear = () => {
    setText('');
    setResults(null);
    setDialect('');
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive':
        return <CheckCircle size={24} color="#10b981" />;
      case 'negative':
        return <XCircle size={24} color="#ef4444" />;
      case 'neutral':
        return <MinusCircle size={24} color="#6b7280" />;
      default:
        return <AlertCircle size={24} color="#f59e0b" />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive':
        return '#10b981';
      case 'negative':
        return '#ef4444';
      case 'neutral':
        return '#6b7280';
      default:
        return '#f59e0b';
    }
  };

  return (
    <Container>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <PageHeader>
          <Title>تحليل المشاعر للنص العربي</Title>
          <Subtitle>
            قم بتحليل مشاعر النص العربي باستخدام الذكاء الاصطناعي
          </Subtitle>
        </PageHeader>

        <ContentGrid>
          {/* Input Section */}
          <InputSection>
            <SectionTitle>
              <Brain size={20} />
              إدخال النص
            </SectionTitle>
            
            <form onSubmit={handleSubmit}>
              <TextArea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="أدخل النص العربي هنا..."
                rows={6}
                dir="rtl"
                lang="ar"
              />
              
              <OptionsRow>
                <Select
                  value={dialect}
                  onChange={(e) => setDialect(e.target.value)}
                  dir="rtl"
                >
                  <option value="">تحديد اللهجة تلقائياً</option>
                  <option value="gulf">اللهجة الخليجية</option>
                  <option value="egyptian">اللهجة المصرية</option>
                  <option value="levantine">اللهجة الشامية</option>
                  <option value="msa">العربية الفصحى</option>
                </Select>
                
                <CheckboxContainer>
                  <input
                    type="checkbox"
                    id="explanation"
                    checked={includeExplanation}
                    onChange={(e) => setIncludeExplanation(e.target.checked)}
                  />
                  <label htmlFor="explanation">تضمين شرح النموذج</label>
                </CheckboxContainer>
              </OptionsRow>
              
              <ButtonRow>
                <SubmitButton
                  type="submit"
                  disabled={sentimentMutation.isLoading || !text.trim()}
                >
                  {sentimentMutation.isLoading ? (
                    <LoadingSpinner size={16} />
                  ) : (
                    <>
                      <Brain size={16} />
                      تحليل المشاعر
                    </>
                  )}
                </SubmitButton>
                
                <ClearButton type="button" onClick={handleClear}>
                  مسح
                </ClearButton>
              </ButtonRow>
            </form>
          </InputSection>

          {/* Results Section */}
          <ResultsSection>
            <SectionTitle>
              <TrendingUp size={20} />
              نتائج التحليل
            </SectionTitle>
            
            {sentimentMutation.isLoading && (
              <LoadingContainer>
                <LoadingSpinner size={32} />
                <LoadingText>جاري تحليل النص...</LoadingText>
              </LoadingContainer>
            )}
            
            {results && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <ResultCard>
                  <ResultHeader>
                    <SentimentIcon>
                      {getSentimentIcon(results.sentiment)}
                    </SentimentIcon>
                    <SentimentInfo>
                      <SentimentLabel>المشاعر</SentimentLabel>
                      <SentimentValue
                        style={{ color: getSentimentColor(results.sentiment) }}
                      >
                        {results.sentiment === 'positive' && 'إيجابي'}
                        {results.sentiment === 'negative' && 'سلبي'}
                        {results.sentiment === 'neutral' && 'محايد'}
                      </SentimentValue>
                    </SentimentInfo>
                  </ResultHeader>
                  
                  <ResultDetails>
                    <DetailItem>
                      <Languages size={16} />
                      <span>اللهجة: {results.dialect}</span>
                    </DetailItem>
                    
                    <DetailItem>
                      <TrendingUp size={16} />
                      <span>مستوى الثقة: {(results.confidence * 100).toFixed(1)}%</span>
                    </DetailItem>
                    
                    <DetailItem>
                      <Clock size={16} />
                      <span>وقت المعالجة: {results.processing_time.toFixed(3)} ثانية</span>
                    </DetailItem>
                  </ResultDetails>
                  
                  {/* Sentiment Chart */}
                  <SentimentChart sentiment={results.sentiment} confidence={results.confidence} />
                  
                  {/* Explanation Panel */}
                  {results.explanation && (
                    <ExplanationPanel explanation={results.explanation} />
                  )}
                </ResultCard>
              </motion.div>
            )}
            
            {!results && !sentimentMutation.isLoading && (
              <EmptyState>
                <Brain size={48} color="#9ca3af" />
                <EmptyText>قم بإدخال نص للحصول على تحليل المشاعر</EmptyText>
              </EmptyState>
            )}
          </ResultsSection>
        </ContentGrid>
      </motion.div>
    </Container>
  );
};

// Styled Components
const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

const PageHeader = styled.div`
  text-align: center;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 12px;
  
  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const Subtitle = styled.p`
  font-size: 1.1rem;
  color: #6b7280;
  max-width: 600px;
  margin: 0 auto;
`;

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 30px;
  }
`;

const Section = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
`;

const InputSection = styled(Section)``;

const ResultsSection = styled(Section)``;

const SectionTitle = styled.h2`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 20px;
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 16px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  font-family: 'Noto Sans Arabic', sans-serif;
  resize: vertical;
  transition: border-color 0.2s;
  
  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
  
  &::placeholder {
    color: #9ca3af;
  }
`;

const OptionsRow = styled.div`
  display: flex;
  gap: 16px;
  margin: 16px 0;
  align-items: center;
  
  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
  }
`;

const Select = styled.select`
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 0.9rem;
  background: white;
  flex: 1;
  
  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
`;

const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  
  input[type="checkbox"] {
    width: 16px;
    height: 16px;
  }
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 20px;
`;

const Button = styled.button`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SubmitButton = styled(Button)`
  background: #3b82f6;
  color: white;
  flex: 1;
  
  &:hover:not(:disabled) {
    background: #2563eb;
  }
`;

const ClearButton = styled(Button)`
  background: #f3f4f6;
  color: #374151;
  
  &:hover {
    background: #e5e7eb;
  }
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 40px;
`;

const LoadingText = styled.p`
  color: #6b7280;
  font-size: 1rem;
`;

const ResultCard = styled.div`
  background: #f9fafb;
  border-radius: 8px;
  padding: 20px;
  border: 1px solid #e5e7eb;
`;

const ResultHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
`;

const SentimentIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  background: white;
  border-radius: 50%;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const SentimentInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const SentimentLabel = styled.span`
  font-size: 0.875rem;
  color: #6b7280;
`;

const SentimentValue = styled.span`
  font-size: 1.25rem;
  font-weight: 600;
`;

const ResultDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 20px;
`;

const DetailItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  color: #4b5563;
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 60px 20px;
  color: #9ca3af;
`;

const EmptyText = styled.p`
  font-size: 1.1rem;
  text-align: center;
`;

export default SentimentAnalysis;
