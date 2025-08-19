import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import GlobalStyles from './styles/GlobalStyles';
import theme from './styles/theme';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import LandingPage from './pages/LandingPage';
import DocumentationPage from './pages/DocumentationPage';
import DemoPage from './pages/DemoPage';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles />
      <Router>
        <div className="App">
          <Header />
          <main>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/docs" element={<DocumentationPage />} />
              <Route path="/demo" element={<DemoPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
