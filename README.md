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

### Порядок запуска от начала до конца

1. **Подготовка** (на каждой машине: мастер и воркеры): Python 3.13, venv, `pip install -r requirements.txt`, `python scripts/gen_grpc.py`.
2. **Воркеры**: на каждом узле (Kubuntu/Ubuntu/Windows) настроить `config/worker.yaml` (порт) и запустить `python -m scripts.run_worker --config config/worker.yaml`.
3. **Мастер**: на узле мастера в `config/master.yaml` указать список воркеров (host/port), затем запустить `python -m scripts.run_master --config config/master.yaml`.
4. **UI** (по желанию): на любой машине с доступом к мастеру запустить `python -m ui.main_window` (при необходимости задать `CLUSTER_MASTER_ADDR=host:port`).

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
python3.13 -m venv .myvenv
source .myvenv/bin/activate
```

- Установить Python‑зависимости и сгенерировать gRPC‑код:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python scripts/gen_grpc.py
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
python -m venv .myvenv
.myvenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts\gen_grpc.py
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

**Сразу после** `pip install -r requirements.txt` выполните генерацию (на каждой машине, где запускаете мастер/воркер/UI):

**Linux (Kubuntu 25 / Ubuntu 24):**

```bash
cd /path/to/chaboss_cluster   # замените на ваш путь к проекту
source .myvenv/bin/activate
python scripts/gen_grpc.py
```

**Windows 10:**

```powershell
cd C:\path\to\chaboss_cluster
.myvenv\Scripts\activate
python scripts\gen_grpc.py
```

В каталоге `cluster_core/grpc/` появятся `cluster_pb2.py` и `cluster_pb2_grpc.py`.  
Без этого шага запуск мастера, воркеров и UI приведёт к ошибке импорта.

---

### 4. Настройка и запуск воркеров (Kubuntu 25, Ubuntu 24, Windows 10)

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

   - **Linux (Kubuntu 25 / Ubuntu 24):**

   ```bash
   cd /path/to/chaboss_cluster
   source .myvenv/bin/activate
   python -m scripts.run_worker --config config/worker.yaml
   ```

   - **Windows 10:**

   ```powershell
   cd C:\chaboss_cluster
   .myvenv\Scripts\activate
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
cd /path/to/chaboss_cluster
source .myvenv/bin/activate
python -m scripts.run_master --config config/master.yaml
```

Мастер:
- поднимет gRPC‑сервер (admin API, воркеры),
- запустит **OpenAI-совместимый HTTP API** на порту `8055` (см. `http_listen_port` в конфиге),
- подключится к воркерам из `master.yaml`, выполнит `GetStatus` и сохранит данные в реестре.

**OpenAI HTTP API** (для OpenCLaw и др.):
- `GET /v1/models` — список моделей (последняя загруженная через UI/LoadModel),
- `POST /v1/chat/completions` — чат (пока заглушка; далее — pipeline по воркерам),
- `POST /v1/completions` — текстовое завершение (заглушка),
- Опционально в `master.yaml`: `openai_api_key: "sk-..."` — проверка заголовка `Authorization: Bearer <key>`.

---

### 6. Запуск PyQt6 UI для мониторинга

На машине мастера или любой другой с доступом к мастеру по сети:

```bash
cd /path/to/chaboss_cluster
source .myvenv/bin/activate
python -m ui.main_window
```

По умолчанию UI подключается к мастеру по адресу `127.0.0.1:60051`.  
Другой адрес задаётся переменной окружения: `CLUSTER_MASTER_ADDR=host:port`.

UI каждые 3 с опрашивает мастер (ListWorkers) и показывает таблицу воркеров: ID, статус, CPU, RAM, GPUs.

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

---

### 8. Что уже сделано и что осталось по плану

**Уже сделано:**
- gRPC‑протокол master ↔ worker (GetStatus, InitShard, RunStage, HealthStream); в StageRequest передаются model_id и shard_id.
- Воркеры: сбор ресурсов (CPU/RAM/GPU), приём шардов (inline_blob, shared_path, hf), сборка BERT‑слоёв из шарда (BertLayer), RunStage с реальным forward по модулю или identity.
- Мастер: реестр воркеров, HealthStream + автопереподключение с backoff, загрузка модели с HF и рассылка шардов (LoadModel); для BERT‑подобных — разбиение по слоям (encoder.layer.i), иначе round‑robin; run_pipeline с передачей model_id/shard_id.
- Admin API: ListWorkers, LoadModel.
- OpenAI‑совместимый HTTP API на порту 8055: /v1/models, /v1/chat/completions, /v1/completions; при загруженной модели chat/completions: токенизатор (кэш по model_id) → эмбеддинги (BERT) → run_pipeline → ответ с учётом выхода энкодера; в usage возвращаются prompt_tokens.
- PyQt6 UI: вкладки Воркеры/Настройки, конфиг воркеров (IP, port, key), модель HF, Скан/Старт, автосохранение настроек; кнопка **«Применить конфиг на мастере»** — отправка списка воркеров на мастер (UpdateWorkersConfig) без перезапуска.

**Осталось по плану:**

1. **Расширение forward на воркерах**  
   Реализован forward для BERT‑подобных (encoder.layer): на воркере собирается Sequential из BertLayer по state_dict шарда. Осталось: поддержка других архитектур (GPT‑2, LLaMA и т.д.), разбиение по слоям для них.

2. **Токенизация и декодирование в HTTP API**  
   Сделано: токенизатор и слой эмбеддингов (BERT) на мастере (кэш по model_id), input_ids → эмбеддинги → run_pipeline; в ответе — форма выхода энкодера, usage.prompt_tokens. Осталось: декодер до текста (для BERT нет генерации; для GPT‑стиля — LM head и decode).

3. **Backend’ы accelerate / deepspeed**  
   Реализовать альтернативные варианты шардирования и инференса (accelerate для точности, deepspeed для скорости), выбор в конфиге или по модели.

4. **Оптимизация передачи тензоров**  
   Варианты: общий storage (NFS/SMB) для весов вместо inline_blob; прямые каналы воркер↔воркер; опционально torch.distributed.rpc / TensorPipe для быстрого обмена тензорами в локальной сети.

5. **Управление шардами по требованию**  
   Выгрузка/загрузка шардов по запросу (lazy load/unload), освобождение VRAM при простое, повторный cold start при смене модели.

6. ~~**Применение конфига воркеров из UI на мастере**~~ — сделано: RPC UpdateWorkersConfig, кнопка «Применить конфиг на мастере» в UI.

7. **Docker и оркестрация**  
   Образы для мастера и воркеров (с поддержкой GPU), docker‑compose, при необходимости — манифесты для Kubernetes.

