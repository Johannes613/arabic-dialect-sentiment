#!/bin/bash

# Arabic Dialect Sentiment Analysis Project Setup Script
# This script sets up the complete project environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found (✓)"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check Node.js version
check_node_version() {
    if command_exists node; then
        NODE_VERSION=$(node --version)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1 | tr -d 'v')
        
        if [ "$NODE_MAJOR" -ge 16 ]; then
            print_success "Node.js $NODE_VERSION found (✓)"
            return 0
        else
            print_error "Node.js 16+ required, found $NODE_VERSION"
            return 1
        fi
    else
        print_error "Node.js not found"
        return 1
    fi
}

# Function to check Docker
check_docker() {
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker found: $DOCKER_VERSION (✓)"
        return 0
    else
        print_warning "Docker not found (optional for development)"
        return 1
    fi
}

# Function to create virtual environment
create_virtual_env() {
    print_status "Creating Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment activated"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy pre-commit
    print_success "Development dependencies installed"
}

# Function to install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    if [ -d "webapp/frontend" ]; then
        cd webapp/frontend
        
        # Install dependencies
        npm install
        
        print_success "Node.js dependencies installed"
        cd ../..
    else
        print_warning "Frontend directory not found, skipping Node.js setup"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    # Create data directories
    mkdir -p data/{raw,processed,external}
    
    # Create model directories
    mkdir -p models/{baselines,dapt,fine_tuned}
    
    # Create log directories
    mkdir -p logs/{baselines,dapt,fine_tuning,tensorboard}
    
    # Create results directory
    mkdir -p results
    
    # Create docker directory
    mkdir -p docker/{grafana/{dashboards,datasources}}
    
    print_success "Project directories created"
}

# Function to download sample data
download_sample_data() {
    print_status "Setting up sample data..."
    
    # Create sample sentiment dataset
    cat > data/raw/sample_sentiment_dataset.csv << 'EOF'
text,label,dialect
هذا مطعم رائع! الطعام لذيذ والخدمة ممتازة,positive,gulf
الفيلم كان ممل جداً، لا أنصح بمشاهدته,negative,egyptian
الطقس اليوم عادي، ليس بارد ولا حار,neutral,msa
شلونك؟ شخبارك؟ عندي مشكلة في السيارة,negative,gulf
مش عارف إزاي أحل المشكلة دي، محتاج مساعدة,negative,egyptian
EOF
    
    print_success "Sample data created"
}

# Function to setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    if command_exists pre-commit; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "Pre-commit not available, skipping"
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env.example << 'EOF'
# Arabic Dialect Sentiment Analysis Environment Variables

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Model Configuration
MODEL_CACHE_DIR=models
MODEL_DEVICE=auto

# Database Configuration (optional)
DATABASE_URL=postgresql://arabic_user:arabic_password@localhost:5432/arabic_sentiment

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# External Services
WANDB_PROJECT=arabic-dialect-sentiment
WANDB_ENTITY=

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
EOF
    
    # Copy to .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Created .env file from template. Please update with your values."
    fi
    
    print_success "Environment configuration created"
}

# Function to create docker configuration files
create_docker_configs() {
    print_status "Creating Docker configuration files..."
    
    # Create nginx configuration
    cat > docker/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    upstream backend {
        server backend:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;
        }
        
        location /api/ {
            proxy_pass http://backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    # Create startup script
    cat > docker/start.sh << 'EOF'
#!/bin/bash

# Start nginx
nginx -g "daemon off;" &

# Start backend
cd /app
python -m uvicorn src.webapp.backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for all background processes
wait
EOF
    
    # Create Prometheus configuration
    cat > docker/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arabic-sentiment-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
EOF
    
    print_success "Docker configuration files created"
}

# Function to create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Create run script
    cat > scripts/run_dev.sh << 'EOF'
#!/bin/bash

# Development environment startup script

echo "Starting Arabic Dialect Sentiment Analysis development environment..."

# Start backend
echo "Starting backend..."
cd "$(dirname "$0")/.."
source venv/bin/activate
python -m uvicorn src.webapp.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend..."
cd webapp/frontend
npm start &
FRONTEND_PID=$!

# Wait for processes
echo "Development environment started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
EOF
    
    # Create test script
    cat > scripts/run_tests.sh << 'EOF'
#!/bin/bash

# Test runner script

echo "Running tests for Arabic Dialect Sentiment Analysis..."

cd "$(dirname "$0")/.."
source venv/bin/activate

# Run Python tests
echo "Running Python tests..."
python -m pytest tests/ -v --cov=src --cov-report=html

# Run frontend tests (if available)
if [ -d "webapp/frontend" ]; then
    echo "Running frontend tests..."
    cd webapp/frontend
    npm test -- --watchAll=false
    cd ../..
fi

echo "Tests completed!"
EOF
    
    # Make scripts executable
    chmod +x scripts/run_dev.sh scripts/run_tests.sh
    
    print_success "Development scripts created"
}

# Function to display next steps
show_next_steps() {
    echo
    print_success "Project setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo
    echo "2. Update the .env file with your configuration"
    echo
    echo "3. Start development environment:"
    echo "   ./scripts/run_dev.sh"
    echo
    echo "4. Run tests:"
    echo "   ./scripts/run_tests.sh"
    echo
    echo "5. Start with Docker (optional):"
    echo "   docker-compose up -d"
    echo
    echo "6. Access the application:"
    echo "   - Backend API: http://localhost:8000"
    echo "   - Frontend: http://localhost:3000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo
    echo "Happy coding! 🚀"
}

# Main setup function
main() {
    echo "=========================================="
    echo "Arabic Dialect Sentiment Analysis Setup"
    echo "=========================================="
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! check_python_version; then
        print_error "Python 3.8+ is required"
        exit 1
    fi
    
    if ! check_node_version; then
        print_error "Node.js 16+ is required"
        exit 1
    fi
    
    check_docker
    
    # Create project structure
    create_directories
    
    # Setup Python environment
    create_virtual_env
    install_python_deps
    
    # Setup Node.js environment
    install_node_deps
    
    # Setup project configuration
    create_env_file
    create_docker_configs
    create_dev_scripts
    
    # Setup development tools
    setup_pre_commit
    
    # Setup sample data
    download_sample_data
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"
