#!/bin/bash
#
# CognitionOS V3 Quick Start Script
#
# Quick setup for local development
#

set -e

echo "🚀 CognitionOS V3 Quick Start"
echo "=============================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys:"
    echo "   - LLM_OPENAI_API_KEY (OpenAI)"
    echo "   - LLM_ANTHROPIC_API_KEY (Anthropic)"
    echo ""
    read -p "Press Enter after updating .env..."
fi

# Start services
echo "Starting services with Docker Compose..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

# Check health
echo ""
echo "🏥 Checking service health..."
if curl -s -f http://localhost:8100/health > /dev/null 2>&1; then
    echo "✅ V3 API is healthy"
else
    echo "⚠️  V3 API is still initializing..."
fi

# Display information
echo ""
echo "✅ CognitionOS is running!"
echo ""
echo "📚 Quick Links:"
echo "   V3 API:          http://localhost:8100"
echo "   API Docs:        http://localhost:8100/docs"
echo "   API Gateway:     http://localhost:8000"
echo "   Frontend:        http://localhost:3000"
echo "   RabbitMQ Admin:  http://localhost:15672 (guest/guest)"
echo ""
echo "📖 Documentation:"
echo "   Phase 2 Guide:   docs/PHASE_2_IMPLEMENTATION.md"
echo "   V3 Architecture: docs/v3/clean_architecture.md"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:       docker-compose logs -f api-v3"
echo "   Stop services:   docker-compose down"
echo "   Restart:         docker-compose restart api-v3"
echo ""
echo "🎯 Try it out:"
echo '   curl http://localhost:8100/health'
echo ""
