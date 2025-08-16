import axios from 'axios';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response.data;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          throw new Error(data.detail || 'بيانات غير صحيحة');
        case 401:
          throw new Error('غير مصرح لك بالوصول');
        case 403:
          throw new Error('ممنوع الوصول');
        case 404:
          throw new Error('المورد غير موجود');
        case 422:
          throw new Error(data.detail || 'بيانات التحقق فشلت');
        case 500:
          throw new Error('خطأ في الخادم');
        case 503:
          throw new Error('الخدمة غير متاحة حالياً');
        default:
          throw new Error(data.detail || `خطأ في الخادم (${status})`);
      }
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('لا يمكن الاتصال بالخادم');
    } else {
      // Something else happened
      throw new Error('حدث خطأ غير متوقع');
    }
  }
);

// Health check
export const checkHealth = async () => {
  return api.get('/health');
};

// Sentiment Analysis
export const analyzeSentiment = async (data) => {
  return api.post('/analyze', data);
};

// Batch Sentiment Analysis
export const analyzeBatchSentiment = async (data) => {
  return api.post('/analyze/batch', data);
};

// Text Preprocessing
export const preprocessText = async (text) => {
  const formData = new FormData();
  formData.append('text', text);
  
  return api.post('/preprocess', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// File Upload
export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  return api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Get Available Models
export const getAvailableModels = async () => {
  return api.get('/models/available');
};

// Get Supported Dialects
export const getSupportedDialects = async () => {
  return api.get('/dialects/supported');
};

// Example Arabic texts for testing
export const getExampleTexts = () => [
  {
    text: "هذا مطعم رائع! الطعام لذيذ والخدمة ممتازة",
    dialect: "gulf",
    expected: "positive",
    description: "Positive review about a restaurant"
  },
  {
    text: "الفيلم كان ممل جداً، لا أنصح بمشاهدته",
    dialect: "egyptian",
    expected: "negative",
    description: "Negative movie review"
  },
  {
    text: "الطقس اليوم عادي، ليس بارد ولا حار",
    dialect: "msa",
    expected: "neutral",
    description: "Neutral weather description"
  },
  {
    text: "شلونك؟ شخبارك؟ عندي مشكلة في السيارة",
    dialect: "gulf",
    expected: "negative",
    description: "Gulf dialect with car problem"
  },
  {
    text: "مش عارف إزاي أحل المشكلة دي، محتاج مساعدة",
    dialect: "egyptian",
    expected: "negative",
    description: "Egyptian dialect asking for help"
  }
];

// Utility functions
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

export const formatProcessingTime = (time) => {
  if (time < 1) {
    return `${(time * 1000).toFixed(0)}ms`;
  }
  return `${time.toFixed(3)}s`;
};

export const getSentimentEmoji = (sentiment) => {
  switch (sentiment?.toLowerCase()) {
    case 'positive':
      return '😊';
    case 'negative':
      return '😞';
    case 'neutral':
      return '😐';
    default:
      return '🤔';
  }
};

export const getSentimentColor = (sentiment) => {
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

export const getDialectName = (dialectCode) => {
  const dialectMap = {
    'gulf': 'اللهجة الخليجية',
    'egyptian': 'اللهجة المصرية',
    'levantine': 'اللهجة الشامية',
    'msa': 'العربية الفصحى',
    'unknown': 'غير معروف'
  };
  
  return dialectMap[dialectCode] || dialectCode;
};

// Error handling utilities
export const isNetworkError = (error) => {
  return !error.response && error.request;
};

export const isServerError = (error) => {
  return error.response && error.response.status >= 500;
};

export const isClientError = (error) => {
  return error.response && error.response.status >= 400 && error.response.status < 500;
};

// Retry logic for failed requests
export const retryRequest = async (requestFn, maxRetries = 3, delay = 1000) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn();
    } catch (error) {
      if (i === maxRetries - 1) {
        throw error;
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
    }
  }
};

// Batch processing with progress tracking
export const processBatchWithProgress = async (texts, onProgress) => {
  const results = [];
  const total = texts.length;
  
  for (let i = 0; i < total; i++) {
    try {
      const result = await analyzeSentiment({ text: texts[i] });
      results.push(result);
      
      // Call progress callback
      if (onProgress) {
        onProgress({
          current: i + 1,
          total,
          percentage: ((i + 1) / total) * 100,
          result
        });
      }
      
      // Small delay to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      results.push({
        text: texts[i],
        error: error.message,
        success: false
      });
    }
  }
  
  return results;
};

export default api;
