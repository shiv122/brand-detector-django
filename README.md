# Detector Backend - Django

Logo Detection API built with Django, MySQL, and YOLO models.

## Prerequisites

- Python 3.11+
- MySQL 8.0+
- [uv](https://github.com/astral-sh/uv) package manager
- YOLO model weights in `weights/` directory

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

### 3. Create MySQL Database

```sql
CREATE DATABASE detector_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. Run Migrations

```bash
uv run python manage.py makemigrations
uv run python manage.py migrate
```

### 5. Copy Model Weights

```bash
# Copy your YOLO weights to backend_new/weights/
cp -r ../backend/weights/* weights/
```

### 6. Run Server

```bash
uv run python manage.py runserver
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /api/v1/health/` - API health check

### Detection
- `GET /api/v1/detection/` - Detection API root
- `GET /api/v1/detection/health/` - Detection service health
- `GET /api/v1/detection/device/` - Device information
- `GET /api/v1/detection/config/` - Get configuration
- `POST /api/v1/detection/config/update/` - Update configuration
- `GET /api/v1/detection/weights/` - Get available weights
- `POST /api/v1/detection/weights/switch/` - Switch weight model
- `POST /api/v1/detection/images/detect/` - Detect logos in images
- `POST /api/v1/detection/video/detect/` - Detect logos in video
- `GET /api/v1/detection/session/{session_id}/summary/` - Get session summary

### Classification
- `GET /api/v1/classification/` - Classification API root
- `GET /api/v1/classification/health/` - Classification service health
- `GET /api/v1/classification/weights/` - Get classification weights
- `POST /api/v1/classification/weights/switch/` - Switch classification weight
- `POST /api/v1/classification/images/classify/` - Classify images

## Environment Variables

See `.env.example` for all available options:

- `SECRET_KEY` - Django secret key
- `DEBUG` - Debug mode (True/False)
- `DB_NAME` - MySQL database name
- `DB_USER` - MySQL username
- `DB_PASSWORD` - MySQL password
- `DB_HOST` - MySQL host
- `DB_PORT` - MySQL port
- `CORS_ALLOWED_ORIGINS` - Allowed CORS origins

## Project Structure

```
backend_new/
├── apps/
│   ├── api/v1/controllers/    # API controllers (Laravel-style)
│   ├── core/models/           # Database models
│   ├── services/              # Business logic services
│   └── utils/                 # Helper functions
├── config/                    # Configuration
├── detector/                  # Django project settings
└── weights/                   # YOLO model weights
```

## Development

### Run Migrations
```bash
uv run python manage.py makemigrations
uv run python manage.py migrate
```

### Create Superuser
```bash
uv run python manage.py createsuperuser
```

### Access Admin Panel
Visit `http://localhost:8000/admin/` after creating superuser

## Notes

- Uses PyMySQL for MySQL connectivity (no compilation required)
- All services use dependency injection
- Direct model usage (no repository pattern)
- Thin controllers delegate to service handlers
- API versioning: `/api/v1/`
# brand-detector-django
# brand-detector-django
# brand-detector-django
