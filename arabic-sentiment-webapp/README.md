# Arabic Sentiment Analysis Web Application

A production-level React web application for Arabic Dialect Sentiment Analysis, featuring a modern design inspired by Vercel's documentation style.

## Features

- **Modern UI/UX**: Clean, responsive design with smooth animations
- **Interactive Demo**: Real-time sentiment analysis demonstration with mock data
- **Comprehensive Documentation**: Detailed guides and API references
- **Responsive Design**: Mobile-first approach with desktop optimization
- **Component-Based Architecture**: Modular, maintainable code structure

## Tech Stack

- **Frontend**: React 18, JavaScript
- **Styling**: Styled Components with custom theme system
- **Animations**: Framer Motion for smooth transitions
- **Icons**: Lucide React for consistent iconography
- **Routing**: React Router for navigation
- **Build Tool**: Create React App

## Project Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Header.js          # Navigation header
│   │   └── Footer.js          # Site footer
│   ├── ui/                    # Reusable UI components
│   └── sections/              # Page sections
├── pages/
│   ├── LandingPage.js         # Home page
│   ├── DocumentationPage.js   # Documentation
│   └── DemoPage.js           # Interactive demo
├── styles/
│   ├── GlobalStyles.js        # Global CSS reset and base styles
│   └── theme.js              # Design system tokens
├── hooks/                     # Custom React hooks
├── utils/                     # Utility functions
└── App.js                     # Main application component
```

## Getting Started

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Johannes613/arabic-dialect-sentiment.git
   cd arabic-dialect-sentiment/arabic-sentiment-webapp
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

## Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App (not recommended)

## Design System

The application uses a comprehensive design system with:

- **Color Palette**: Primary, secondary, accent, and semantic colors
- **Typography**: Consistent font scales and weights
- **Spacing**: 8-point grid system
- **Shadows**: Multiple elevation levels
- **Border Radius**: Consistent corner rounding
- **Breakpoints**: Responsive design breakpoints

## Components

### Layout Components

- **Header**: Fixed navigation with mobile menu
- **Footer**: Multi-column footer with links and social media

### Page Components

- **LandingPage**: Hero section, features, and call-to-action
- **DocumentationPage**: Sidebar navigation with content sections
- **DemoPage**: Interactive sentiment analysis interface

### UI Components

- **Buttons**: Primary, secondary, and disabled states
- **Cards**: Feature cards, result cards, and stat cards
- **Forms**: Text areas and input fields
- **Navigation**: Sidebar navigation and breadcrumbs

## Responsive Design

The application is fully responsive with:

- Mobile-first approach
- Breakpoint-based layouts
- Touch-friendly interactions
- Optimized typography scaling

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance Features

- Lazy loading of components
- Optimized animations with Framer Motion
- Efficient re-renders with React hooks
- Minimal bundle size

## Future Enhancements

- **Backend Integration**: Connect to FastAPI backend
- **Real-time Analysis**: Live sentiment analysis
- **User Authentication**: User accounts and history
- **Advanced Analytics**: Detailed performance metrics
- **Multi-language Support**: Additional language interfaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Arabic Dialect Sentiment Analysis system.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo page

---

Built with ❤️ using React and modern web technologies.
