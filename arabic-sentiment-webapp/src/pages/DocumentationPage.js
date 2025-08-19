import React, { useState } from 'react';
import styled from 'styled-components';
import { Play, Github, ExternalLink, Copy, Check } from 'lucide-react';

const PageContainer = styled.div`
  padding-top: 64px;
  display: flex;
  min-height: calc(100vh - 64px);
`;

const Sidebar = styled.aside`
  width: 280px;
  background: ${props => props.theme.colors.vercel.gray[950]};
  border-right: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing[6]} 0;
  position: sticky;
  top: 64px;
  height: calc(100vh - 64px);
  overflow-y: auto;

  @media (max-width: ${props => props.theme.breakpoints.lg}) {
    display: none;
  }
`;

const SidebarContent = styled.div`
  padding: 0 ${props => props.theme.spacing[4]};
`;

const SidebarTitle = styled.h3`
  font-size: ${props => props.theme.fontSizes.lg};
  font-weight: ${props => props.theme.fontWeights.semibold};
  margin-bottom: ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.text.primary};
`;

const SidebarNav = styled.nav`
  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  li {
    margin-bottom: ${props => props.theme.spacing[1]};
  }
`;

const SidebarLink = styled.a`
  display: block;
  padding: ${props => props.theme.spacing[2]} ${props => props.theme.spacing[3]};
  color: ${props => props.theme.colors.text.secondary};
  text-decoration: none;
  border-radius: ${props => props.theme.borderRadius.md};
  transition: all 0.2s ease;
  font-size: ${props => props.theme.fontSizes.sm};

  &:hover {
    background: ${props => props.theme.colors.vercel.gray[900]};
    color: ${props => props.theme.colors.text.primary};
  }

  &.active {
    background: ${props => props.theme.colors.primary};
    color: ${props => props.theme.colors.text.inverse};
  }
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${props => props.theme.spacing[8]} ${props => props.theme.spacing[6]};
  max-width: calc(100% - 280px);
`;

const Section = styled.section`
  margin-bottom: ${props => props.theme.spacing[12]};
  scroll-margin-top: 80px;
`;

const SectionTitle = styled.h2`
  font-size: ${props => props.theme.fontSizes['3xl']};
  margin-bottom: ${props => props.theme.spacing[6]};
  color: ${props => props.theme.colors.text.primary};
  border-bottom: 2px solid ${props => props.theme.colors.border};
  padding-bottom: ${props => props.theme.spacing[3]};
`;

const SectionSubtitle = styled.h3`
  font-size: ${props => props.theme.fontSizes.xl};
  margin: ${props => props.theme.spacing[6]} 0 ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.text.primary};
`;

const CodeBlockContainer = styled.div`
  position: relative;
  margin: ${props => props.theme.spacing[4]} 0;
`;

const CodeBlock = styled.pre`
  background: ${props => props.theme.colors.vercel.gray[900]};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing[4]};
  overflow-x: auto;
  font-family: ${props => props.theme.fonts.mono};
  font-size: ${props => props.theme.fontSizes.sm};
  margin: 0;
`;

const CopyButton = styled.button`
  position: absolute;
  top: ${props => props.theme.spacing[2]};
  right: ${props => props.theme.spacing[2]};
  background: ${props => props.theme.colors.vercel.gray[800]};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.sm};
  padding: ${props => props.theme.spacing[1]} ${props => props.theme.spacing[2]};
  color: ${props => props.theme.colors.text.secondary};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[1]};
  font-size: ${props => props.theme.fontSizes.xs};

  &:hover {
    background: ${props => props.theme.colors.vercel.gray[700]};
    color: ${props => props.theme.colors.text.primary};
  }

  &:active {
    transform: scale(0.95);
  }
`;

const InlineCode = styled.code`
  background: ${props => props.theme.colors.vercel.gray[900]};
  padding: ${props => props.theme.spacing[1]} ${props => props.theme.spacing[2]};
  border-radius: ${props => props.theme.borderRadius.sm};
  font-family: ${props => props.theme.fonts.mono};
  font-size: ${props => props.theme.fontSizes.sm};
`;

const InfoBox = styled.div`
  background: ${props => props.background || props.theme.colors.vercel.gray[900]};
  border: 1px solid ${props => props.border || props.theme.colors.border};
  border-left: 4px solid ${props => props.accent || props.theme.colors.accent};
  padding: ${props => props.theme.spacing[4]};
  border-radius: ${props => props.theme.borderRadius.md};
  margin: ${props => props.theme.spacing[4]} 0;
`;

const InfoBoxTitle = styled.h4`
  margin: 0 0 ${props => props.theme.spacing[2]} 0;
  color: ${props => props.theme.colors.text.primary};
  font-weight: ${props => props.theme.fontWeights.semibold};
`;

const InfoBoxContent = styled.div`
  color: ${props => props.theme.colors.text.secondary};
  line-height: 1.6;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin: ${props => props.theme.spacing[4]} 0;
  background: ${props => props.theme.colors.vercel.gray[900]};
  border-radius: ${props => props.theme.borderRadius.md};
  overflow: hidden;
  border: 1px solid ${props => props.theme.colors.border};
`;

const TableHeader = styled.th`
  background: ${props => props.theme.colors.vercel.gray[950]};
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  text-align: left;
  font-weight: ${props => props.theme.fontWeights.semibold};
  color: ${props => props.theme.colors.text.primary};
`;

const TableCell = styled.td`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  color: ${props => props.theme.colors.text.secondary};
`;

const Button = styled.a`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  background: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.text.inverse};
  text-decoration: none;
  border-radius: ${props => props.theme.borderRadius.md};
  font-weight: ${props => props.theme.fontWeights.medium};
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.vercel.gray[200]};
    transform: translateY(-1px);
  }
`;

const CopyableCodeBlock = ({ children, language = 'text' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  return (
    <CodeBlockContainer>
      <CodeBlock>{children}</CodeBlock>
      <CopyButton onClick={handleCopy}>
        {copied ? (
          <>
            <Check size={12} />
            Copied!
          </>
        ) : (
          <>
            <Copy size={12} />
            Copy
          </>
        )}
      </CopyButton>
    </CodeBlockContainer>
  );
};

const DocumentationPage = () => {
  const [activeSection, setActiveSection] = useState('getting-started');

  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <PageContainer>
      <Sidebar>
        <SidebarContent>
          <SidebarTitle>Documentation</SidebarTitle>
          <SidebarNav>
            <ul>
              <li>
                <SidebarLink 
                  href="#getting-started" 
                  className={activeSection === 'getting-started' ? 'active' : ''}
                  onClick={() => scrollToSection('getting-started')}
                >
                  Getting Started
                </SidebarLink>
              </li>
              <li>
                <SidebarLink 
                  href="#installation" 
                  className={activeSection === 'installation' ? 'active' : ''}
                  onClick={() => scrollToSection('installation')}
                >
                  Installation
                </SidebarLink>
              </li>
              <li>
                <SidebarLink 
                  href="#quick-start" 
                  className={activeSection === 'quick-start' ? 'active' : ''}
                  onClick={() => scrollToSection('quick-start')}
                >
                  Quick Start
                </SidebarLink>
              </li>
              <li>
                <SidebarLink 
                  href="#api-reference" 
                  className={activeSection === 'api-reference' ? 'active' : ''}
                  onClick={() => scrollToSection('api-reference')}
                >
                  API Reference
                </SidebarLink>
              </li>
              <li>
                <SidebarLink 
                  href="#examples" 
                  className={activeSection === 'examples' ? 'active' : ''}
                  onClick={() => scrollToSection('examples')}
                >
                  Examples
                </SidebarLink>
              </li>
              <li>
                <SidebarLink 
                  href="#deployment" 
                  className={activeSection === 'deployment' ? 'active' : ''}
                  onClick={() => scrollToSection('deployment')}
                >
                  Deployment
                </SidebarLink>
              </li>
            </ul>
          </SidebarNav>
        </SidebarContent>
      </Sidebar>

      <MainContent>
        <Section id="getting-started">
          <SectionTitle>Getting Started</SectionTitle>
          <p>
            Welcome to the Arabic Dialect Sentiment Analysis documentation. This guide will help you 
            understand how to use our enhanced MARBERT model for sentiment analysis in Arabic dialects.
          </p>
          
          <InfoBox>
            <InfoBoxTitle>What is Arabic Sentiment Analysis?</InfoBoxTitle>
            <InfoBoxContent>
              Our system provides advanced sentiment analysis capabilities for Arabic dialects, 
              particularly Gulf Arabic. It uses a domain-adapted transformer model (MARBERT) 
              that achieves 88% accuracy and 86% Macro F1 score on the ASTD dataset.
            </InfoBoxContent>
          </InfoBox>

          <SectionSubtitle>Key Features</SectionSubtitle>
          <ul>
            <li><strong>Enhanced MARBERT:</strong> Domain-adapted transformer model optimized for Arabic dialects</li>
            <li><strong>Class Balancing:</strong> Advanced techniques to handle class imbalance</li>
            <li><strong>Data Augmentation:</strong> Arabic-specific text augmentation strategies</li>
            <li><strong>Production Ready:</strong> FastAPI backend with React frontend</li>
            <li><strong>Comprehensive Evaluation:</strong> Detailed performance metrics and analysis</li>
          </ul>
        </Section>

        <Section id="installation">
          <SectionTitle>Installation</SectionTitle>
          <p>
            Follow these steps to set up the Arabic Sentiment Analysis system in your environment.
          </p>

          <SectionSubtitle>Prerequisites</SectionSubtitle>
          <ul>
            <li>Python 3.8 or higher</li>
            <li>PyTorch 1.9+</li>
            <li>Transformers 4.20+</li>
            <li>CUDA-compatible GPU (recommended)</li>
          </ul>

          <SectionSubtitle>Clone the Repository</SectionSubtitle>
          <CopyableCodeBlock language="bash">
{`git clone https://github.com/Johannes613/arabic-dialect-sentiment.git
cd arabic-dialect-sentiment`}
          </CopyableCodeBlock>

          <SectionSubtitle>Install Dependencies</SectionSubtitle>
          <CopyableCodeBlock language="bash">
{`pip install -r requirements.txt`}
          </CopyableCodeBlock>

          <InfoBox>
            <InfoBoxTitle>Note</InfoBoxTitle>
            <InfoBoxContent>
              For the web application specifically, navigate to the webapp directory and install 
              the frontend dependencies: <InlineCode>cd webapp/frontend && npm install</InlineCode>
            </InfoBoxContent>
          </InfoBox>
        </Section>

        <Section id="quick-start">
          <SectionTitle>Quick Start</SectionTitle>
          <p>
            Get up and running with sentiment analysis in just a few minutes.
          </p>

          <SectionSubtitle>Basic Usage</SectionSubtitle>
          <CopyableCodeBlock language="python">
{`from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./models")
tokenizer = AutoTokenizer.from_pretrained("./models")

# Prepare input text
text = "هذا النص رائع جدا"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()

# Map class index to label
labels = ["NEG", "POS", "NEUTRAL", "OBJ"]
sentiment = labels[predicted_class]
confidence = predictions[0][predicted_class].item()

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.3f}")`}
          </CopyableCodeBlock>

          <SectionSubtitle>Using the Web Interface</SectionSubtitle>
          <p>
            For a user-friendly interface, you can use our web application:
          </p>
          <CopyableCodeBlock language="bash">
{`# Start the backend
cd webapp/backend
uvicorn main:app --reload

# Start the frontend (in another terminal)
cd webapp/frontend
npm start`}
          </CopyableCodeBlock>
        </Section>

        <Section id="api-reference">
          <SectionTitle>API Reference</SectionTitle>
          <p>
            The FastAPI backend provides RESTful endpoints for sentiment analysis.
          </p>

          <SectionSubtitle>Endpoints</SectionSubtitle>
          <Table>
            <thead>
              <tr>
                <TableHeader>Endpoint</TableHeader>
                <TableHeader>Method</TableHeader>
                <TableHeader>Description</TableHeader>
                <TableHeader>Parameters</TableHeader>
              </tr>
            </thead>
            <tbody>
              <tr>
                <TableCell><InlineCode>/health</InlineCode></TableCell>
                <TableCell>GET</TableCell>
                <TableCell>Health check endpoint</TableCell>
                <TableCell>None</TableCell>
              </tr>
              <tr>
                <TableCell><InlineCode>/analyze</InlineCode></TableCell>
                <TableCell>POST</TableCell>
                <TableCell>Single text sentiment analysis</TableCell>
                <TableCell>text, dialect (optional)</TableCell>
              </tr>
              <tr>
                <TableCell><InlineCode>/analyze/batch</InlineCode></TableCell>
                <TableCell>POST</TableCell>
                <TableCell>Batch sentiment analysis</TableCell>
                <TableCell>texts array, dialect (optional)</TableCell>
              </tr>
              <tr>
                <TableCell><InlineCode>/preprocess</InlineCode></TableCell>
                <TableCell>POST</TableCell>
                <TableCell>Arabic text preprocessing</TableCell>
                <TableCell>text</TableCell>
              </tr>
            </tbody>
          </Table>

          <SectionSubtitle>Request Format</SectionSubtitle>
          <CopyableCodeBlock language="json">
{`{
  "text": "النص العربي للتحليل",
  "dialect": "gulf"  // optional
}`}
          </CopyableCodeBlock>

          <SectionSubtitle>Response Format</SectionSubtitle>
          <CopyableCodeBlock language="json">
{`{
  "sentiment": "POS",
  "confidence": 0.89,
  "processing_time": 0.045,
  "text": "النص العربي للتحليل",
  "dialect": "gulf"
}`}
          </CopyableCodeBlock>
        </Section>

        <Section id="examples">
          <SectionTitle>Examples</SectionTitle>
          <p>
            Explore practical examples of using the Arabic Sentiment Analysis system.
          </p>

          <SectionSubtitle>Python Client Example</SectionSubtitle>
          <CopyableCodeBlock language="python">
{`import requests
import json

# API endpoint
url = "http://localhost:8000/analyze"

# Sample Arabic texts
texts = [
    "هذا المنتج ممتاز جدا",
    "الخدمة سيئة للغاية",
    "التجربة كانت عادية",
    "أنا محايد في هذا الموضوع"
]

# Analyze each text
for text in texts:
    response = requests.post(url, json={"text": text})
    result = response.json()
    
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("---")`}
          </CopyableCodeBlock>

          <SectionSubtitle>JavaScript/React Example</SectionSubtitle>
          <CopyableCodeBlock language="javascript">
{`import React, { useState } from 'react';

const SentimentAnalyzer = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="أدخل النص العربي هنا..."
        rows={4}
      />
      <button onClick={analyzeSentiment} disabled={loading}>
        {loading ? 'جاري التحليل...' : 'تحليل المشاعر'}
      </button>
      {result && (
        <div>
          <h3>النتيجة: {result.sentiment}</h3>
          <p>الثقة: {result.confidence.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
};`}
          </CopyableCodeBlock>
        </Section>

        <Section id="deployment">
          <SectionTitle>Deployment</SectionTitle>
          <p>
            Deploy the Arabic Sentiment Analysis system to production environments.
          </p>

          <SectionSubtitle>Docker Deployment</SectionSubtitle>
          <p>
            Use our Docker configuration for easy deployment:
          </p>
          <CopyableCodeBlock language="bash">
{`# Build and run with Docker Compose
docker-compose up --build

# Or build individual services
docker build -t arabic-sentiment-backend ./webapp/backend
docker build -t arabic-sentiment-frontend ./webapp/frontend`}
          </CopyableCodeBlock>

          <SectionSubtitle>Environment Variables</SectionSubtitle>
          <CopyableCodeBlock language="bash">
{`# Backend environment variables
MODEL_PATH=./models
MAX_LENGTH=512
BATCH_SIZE=32
DEVICE=cuda  # or cpu

# Frontend environment variables
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=production`}
          </CopyableCodeBlock>

          <SectionSubtitle>Production Considerations</SectionSubtitle>
          <ul>
            <li><strong>Model Optimization:</strong> Consider model quantization for faster inference</li>
            <li><strong>Load Balancing:</strong> Use multiple backend instances for high traffic</li>
            <li><strong>Monitoring:</strong> Implement logging and metrics collection</li>
            <li><strong>Security:</strong> Add authentication and rate limiting</li>
            <li><strong>Scaling:</strong> Use container orchestration (Kubernetes) for large deployments</li>
          </ul>
        </Section>

        <Section>
          <SectionTitle>Next Steps</SectionTitle>
          <p>
            Now that you have a basic understanding, here are some next steps:
          </p>
          <ul>
            <li>Try the live demo to see the system in action</li>
            <li>Explore the GitHub repository for source code and examples</li>
            <li>Join our community discussions for support and feedback</li>
            <li>Contribute to the project by reporting issues or submitting pull requests</li>
          </ul>

          <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <Button href="/demo">
              <Play size={16} />
              Try Demo
            </Button>
            <Button href="https://github.com/Johannes613/arabic-dialect-sentiment" target="_blank">
              <Github size={16} />
              View on GitHub
            </Button>
            <Button href="https://github.com/Johannes613/arabic-dialect-sentiment/issues" target="_blank">
              <ExternalLink size={16} />
              Report Issues
            </Button>
          </div>
        </Section>
      </MainContent>
    </PageContainer>
  );
};

export default DocumentationPage;
