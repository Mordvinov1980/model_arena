#!/bin/bash

cleanup() {
    echo ""
    echo "🛑 Завершение работы..."
    pkill -f "python.*app.py" 2>/dev/null
    pkill -f "ssh.*serveo" 2>/dev/null
    echo "✅ Процессы остановлены"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "🚀 Запуск AI-ARENA..."

# Очистка старых процессов
echo "🛑 Останавливаем старые процессы..."
pkill -f "python.*app.py" 2>/dev/null
pkill -f "ssh.*serveo" 2>/dev/null
sleep 5

# Запуск Ollama
OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=-1 CUDA_VISIBLE_DEVICES=0 ollama serve > /dev/null 2>&1 &
sleep 5

# Запуск AI_ARENA
echo "🚀 Запуск на порту 8080..."
python app.py > app.log 2>&1 &
SERVER_PID=$!

# Ждём загрузки
echo "⏳ Ждём загрузки (10 секунд)..."
sleep 10
if ! curl -s http://localhost:8080 > /dev/null; then
    echo "❌ AI-ARENA не запустилась! Проверь app.log"
    exit 1
fi
echo "✅ AI-ARENA работает!"

TUNNEL_URL="https://tsm-ai.serveousercontent.com"

echo ""
echo "══════════════════════════======═════════════"
echo "🌐 LINK: $TUNNEL_URL"
echo "════════════════════════════════======═══════"
echo ""
echo "🔄 Авто-перезапуск при обрыве соединения"
echo "💡 Двойной Ctrl+C для полной остановки"
echo ""

# Цикл авто-перезапуска туннеля
while true; do
    echo "🔄 Подключение к Serveo... ($(date '+%H:%M:%S'))"
    ssh -o ServerAliveInterval=60 \
        -o StrictHostKeyChecking=no \
        -o ExitOnForwardFailure=yes \
        -R tsm-ai:80:localhost:8080 \
        serveo.net

    echo "⚠️ Соединение разорвано ($(date '+%H:%M:%S'))"
    echo "🔄 Переподключение через 10 секунд..."
    sleep 10
done