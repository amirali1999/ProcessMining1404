# Process Mining Tool

A Django-based process mining application for analyzing and visualizing business processes.

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/amir-saberi/process-mining-tool-inspired-by-disco.git
cd process-mining-tool-inspired-by-disco
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

5. Run the development server:
```bash
python manage.py runserver
```

6. Open your browser and navigate to:
- **Main App**: http://127.0.0.1:8000/
- **Admin Panel**: http://127.0.0.1:8000/admin/

## Features
- Event log preprocessing
- Process discovery (Alpha, Heuristic, Inductive miners)
- Conformance checking
- Process prediction
- User authentication & license management

## Default Login
- Register a new account at `/register/`
- Or access admin panel at `/admin-login/`

## License
This project is for educational purposes.
