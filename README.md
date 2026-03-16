## Chaboss Cluster

Локальный/гибридный кластер для распределённого запуска крупных PyTorch‑моделей  
по схеме «1 мастер → N воркеров» с упором на **модель‑параллелизм**.

Основной стек:
- **Python**: 3.13
- **RPC**: gRPC (HTTP/2, bidi‑stream)
- **DL**: PyTorch (+ планируется `accelerate`/`deepspeed` и др.)
- **UI**: PyQt6

---

### Структура проекта

- `cluster_core/` — ядро кластера:
  - `common/` — общие типы, конфиг.
  - `master/` — логика основного узла (реестр воркеров и т.п.).
  - `worker/` — реализация gRPC‑сервиса воркера.
  - `grpc/` — сгенерированные Python‑файлы из `.proto`.
- `proto/` — gRPC `.proto` (описание протокола master ↔ worker).
- `scripts/run_master.py` — запуск основного узла.
- `scripts/run_worker.py` — запуск воркера.
- `ui/main_window.py` — PyQt6 UI для мониторинга кластера.
- `config/` — базовые YAML‑конфиги мастера и воркера.
- `requirements.txt` — зависимости Python.

---

### 1. Подготовка окружения (Kubuntu 25 / Ubuntu 24, мастер и/или воркер)

- **Установить Python 3.13** (либо системный, либо через `pyenv`).
- Установить необходимые системные пакеты:

```bash
sudo apt update
sudo apt install -y python3-dev python3-venv python3-pip build-essential
sudo apt install -y libgl1  # для PyQt6 и некоторых графических зависимостей
```

- (Рекомендуется) создать виртуальное окружение в корне проекта:

```bash
cd /home/chaboss/work/python/chaboss_cluster
python3.13 -m venv .venv
source .venv/bin/activate
```

- Установить Python‑зависимости:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- Убедиться, что CUDA/драйверы установлены (для GPU‑узлов):
  - NVIDIA: проверка `nvidia-smi`.
  - PyTorch: в Python‑консоли:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
```

---

### 2. Подготовка воркера под Windows 10

На Windows 10 воркер также запускается как независимый gRPC‑сервер.

- Установить:
  - **Python 3.13 (x64)**.
  - Совместимую сборку **PyTorch** с поддержкой CUDA (для NVIDIA)  
    или CPU‑сборку, если GPU не используется.
  - Git (для удобства).

- Клонировать репозиторий проекта на Windows‑машину:

```powershell
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ> C:\chaboss_cluster
cd C:\chaboss_cluster
```

- Создать виртуальное окружение и установить зависимости:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

- При необходимости отредактировать `config/worker.yaml` под Windows‑узел:
  - задать `listen_host` (обычно `"0.0.0.0"`),
  - уникальный `listen_port`,
  - при желании `auth_token`.

- Запустить воркер:

```powershell
python -m scripts.run_worker --config config/worker.yaml
```

Важно: мастер должен иметь сетевой доступ до этого Windows‑узла (порт `listen_port`).

---

### 3. Генерация gRPC‑кода (обязательно до первого запуска)

После любых изменений `proto/cluster.proto` нужно пересобрать Python‑стабы:

```bash
cd /home/chaboss/work/python/chaboss_cluster

python -m grpc_tools.protoc \
  -I proto \
  --python_out=cluster_core/grpc \
  --grpc_python_out=cluster_core/grpc \
  proto/cluster.proto
```

В результате в `cluster_core/grpc/` появятся файлы `cluster_pb2.py` и `cluster_pb2_grpc.py`,  
которые используются модулем воркера и мастера.

На Windows команда аналогична (в активированном виртуальном окружении PowerShell).

---

### 4. Настройка и запуск воркеров (Linux и Windows)

1. **Отредактировать `config/worker.yaml` на каждом воркере**:

   Пример (Linux‑воркер):

   ```yaml
   listen_host: "0.0.0.0"
   listen_port: 60052
   auth_token: null
   ```

   Для каждого воркера:
   - назначьте **уникальный порт** (`listen_port`),
   - при необходимости выставьте `auth_token`.

2. **Запустить воркера**:

   - Linux (Kubuntu/Ubuntu):

   ```bash
   cd /home/chaboss/work/python/chaboss_cluster
   source .venv/bin/activate  # если используете venv
   python -m scripts.run_worker --config config/worker.yaml
   ```

   - Windows 10:

   ```powershell
   cd C:\chaboss_cluster
   .venv\Scripts\activate
   python -m scripts.run_worker --config config\worker.yaml
   ```

   После запуска воркер поднимет gRPC‑сервер и будет готов принимать запросы от мастера.

---

### 5. Настройка и запуск мастера (Kubuntu 25 / Ubuntu 24)

1. **Отредактировать `config/master.yaml` на машине мастера**:

   ```yaml
   listen_host: "0.0.0.0"
   listen_port: 60051

   parallel_backend: "baseline"  # baseline | accelerate | deepspeed

   workers:
     - host: "192.168.1.10"   # Linux‑воркер
       port: 60052
       auth_token: null
     - host: "192.168.1.20"   # Windows‑воркер
       port: 60053
       auth_token: null
   ```

   Укажите реальные IP‑адреса и порты всех воркеров (Linux/Windows), а также  
   выберите начальный backend (`baseline` для первого запуска).

2. **Запустить мастера**:

```bash
cd /home/chaboss/work/python/chaboss_cluster
source .venv/bin/activate  # при использовании виртуального окружения
python -m scripts.run_master --config config/master.yaml
```

Мастер:
- поднимет свой gRPC‑сервер (для последующего admin/external API),
- попробует подключиться к каждому воркеру из `master.yaml`,
- выполнит `GetStatus` и сохранит данные о ресурсах в реестре.

---

### 6. Запуск PyQt6 UI для мониторинга

На машине мастера (или любой другой, имеющей сетевой доступ к мастеру):

```bash
cd /home/chaboss/work/python/chaboss_cluster
source .venv/bin/activate
python -m ui.main_window
```

Сейчас UI выводит таблицу воркеров (каркас), в следующих итерациях будет:
- подключаться к мастеру по admin‑API,
- отображать реальные статусы, ресурсы, загруженные модели и ошибки.

---

### 7. Дальнейшее развитие и оптимизация

- **Модель‑параллелизм**:
  - baseline‑backend: разбиение PyTorch‑модели по слоям (pipeline parallel) между воркерами;
  - добавление backend’ов `accelerate` и `deepspeed`.
- **Быстрый транспорт тензоров**:
  - альтернативный backend на базе `torch.distributed.rpc`/TensorPipe/NCCL внутри локальной сети;
  - прямые каналы воркер↔воркер для pipeline parallel.
- **Холодный старт и веса моделей**:
  - режим, когда воркеры сами скачивают веса с Hugging Face;
  - режим с общим storage (NFS/SMB) и минимальной передачей по сети;
  - управление загрузкой/выгрузкой шардов по требованию.
- **OpenAI‑совместимый API**:
  - HTTP‑слой (`/v1/chat/completions` и др.) поверх мастера,
  - интеграция с внешними клиентами (включая OpenCLaw и аналоги).
- **Docker и оркестрация**:
  - Docker‑образы для мастера и воркеров (с поддержкой GPU),
  - docker‑compose / Kubernetes‑манифесты.

