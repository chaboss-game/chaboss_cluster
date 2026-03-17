#!/usr/bin/env bash
# Запуск GUI Chaboss Cluster из корня проекта.
# Использование: ./run_gui.sh [адрес_мастера]
# Пример: ./run_gui.sh 127.0.0.1:60051
# Переменная CLUSTER_MASTER_ADDR переопределяет адрес мастера.

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".myvenv" ]; then
    source .myvenv/bin/activate
fi

if [ -n "$1" ]; then
    export CLUSTER_MASTER_ADDR="$1"
fi

exec python3 -m ui.main_window
