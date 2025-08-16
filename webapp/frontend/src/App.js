import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import styled from 'styled-components';

// Components
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import SentimentAnalysis from './pages/SentimentAnalysis';
import BatchAnalysis from './pages/BatchAnalysis';
import ModelInfo from './pages/ModelInfo';
import About from './pages/About';

// Styles
import './styles/App.css';
import './styles/RTL.css';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Styled components
const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  direction: rtl; /* RTL support for Arabic */
`;

const MainContent = styled.main`
  flex: 1;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  
  @media (max-width: 768px) {
    padding: 10px;
  }
`;

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppContainer>
          <Header />
          <MainContent>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/analyze" element={<SentimentAnalysis />} />
              <Route path="/batch" element={<BatchAnalysis />} />
              <Route path="/model" element={<ModelInfo />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </MainContent>
          <Footer />
          <Toaster
            position="top-center"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
                direction: 'rtl',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#4ade80',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </AppContainer>
      </Router>
    </QueryClientProvider>
  );
};

export default App;
