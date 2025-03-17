# AI Wetter - Bulgarian Weather Forecast Tool

A web-based application that gathers weather forecast data from multiple sources (ECMWF, Meteoblue, Meteologix) and creates combined reports focused on agricultural needs in Bulgaria.

## Features

- Fetches 14-day weather forecasts from multiple weather data sources
- Allows users to select different locations within Bulgaria
- Provides detailed weather visualizations with interactive charts
- Stores historical forecast data for comparison
- Generates comprehensive weather reports with agricultural recommendations
- Calculates forecast confidence levels based on agreement between sources

## Screenshots

(Screenshots will be added when the application is running)

## Installation

### Local Development

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai_wetter.git
cd ai_wetter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install Playwright browsers:
```bash
python -m playwright install chromium
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit the .env file with your configuration
```

6. Run the tests to verify functionality:
```bash
python -m unittest discover tests
```

7. Run the application:
```bash
python -m app.app --debug
```

8. Open your web browser and navigate to:
```
http://localhost:5000
```

### Deployment on Ubuntu Server

#### Option 1: Using Supervisor

1. Connect to your Ubuntu server:
```bash
ssh username@your_server_ip
```

2. Install required system packages:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx supervisor build-essential

# Install required dependencies for Playwright
sudo apt-get install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
  libpango-1.0-0 libcairo2 libatspi2.0-0
```

3. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_wetter.git
cd ai_wetter
```

4. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Install dependencies:
```bash
pip install -r requirements.txt
pip install gunicorn
```

6. Install Playwright browsers:
```bash
python -m playwright install chromium
```

7. Set up environment variables:
```bash
cp .env.example .env
# Edit the .env file with your configuration
nano .env
```

8. Create logs directory and Supervisor configuration:
```bash
# Create logs directory
mkdir -p ~/ai_wetter/logs

# Create Supervisor configuration
sudo nano /etc/supervisor/conf.d/ai_wetter.conf
```

Add the following content (adjust paths as necessary):
```
[program:ai_wetter]
directory=/home/your_username/ai_wetter
command=/home/your_username/ai_wetter/venv/bin/gunicorn 'app.app:create_app()' --bind=127.0.0.1:8000 --workers=2
autostart=true
autorestart=true
stderr_logfile=/home/your_username/ai_wetter/logs/gunicorn.err.log
stdout_logfile=/home/your_username/ai_wetter/logs/gunicorn.out.log
user=your_username
environment=PATH="/home/your_username/ai_wetter/venv/bin"
```

Make sure to replace `your_username` with your actual username (e.g., `jerry`)

9. Set up Nginx as a reverse proxy:
```bash
sudo nano /etc/nginx/sites-available/ai_wetter
```

Add the following configuration:
```
server {
    listen 80;
    server_name your_domain.com;  # Or your server IP

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

10. Enable the site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ai_wetter /etc/nginx/sites-enabled/
sudo nginx -t  # Test the configuration
sudo systemctl restart nginx
```

11. Start the application with Supervisor:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start ai_wetter
```

12. Your application should now be running at `http://your_domain.com` or `http://your_server_ip`.

#### Option 2: Using Systemd (Alternative)

Follow steps 1-7 from Option 1, then:

8. Create a systemd service file:
```bash
sudo cp systemd/ai_wetter.service.example /etc/systemd/system/ai_wetter.service
sudo nano /etc/systemd/system/ai_wetter.service
```
Edit the file to match your paths and username.

9. Set up Nginx as described in Option 1 (steps 9-10).

10. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai_wetter
sudo systemctl start ai_wetter
```

11. Check the service status:
```bash
sudo systemctl status ai_wetter
```

12. Your application should now be running at `http://your_domain.com` or `http://your_server_ip`.

## Usage

1. Select a location from the dropdown menu
2. Choose the number of forecast days (1-14)
3. Click "Get Forecast" to view weather data
4. Use the navigation buttons to:
   - View forecast history
   - Generate comprehensive reports
   - Access saved reports

## Weather Data Sources

The application uses the following data sources:
- **ECMWF** (European Centre for Medium-Range Weather Forecasts) - Known for accuracy in long-range forecasts
- **Meteoblue** - Provides high-resolution local weather forecasts
- **Meteologix** - Offers multi-model comparisons (currently a placeholder for future implementation)

## Project Structure

- `app/` - Main application directory
  - `app.py` - Application entry point
  - `data_sources/` - Weather data source implementations
  - `models/` - Data models for locations and forecasts
  - `services/` - Business logic services
  - `ui/` - Web interface components
  - `utils/` - Utility functions
  - `templates/` - HTML templates
  - `static/` - Static files (CSS, JS)
- `data/` - Data storage
  - `history/` - Historical forecast data
  - `reports/` - Saved reports
  - `logs/` - Application logs

## API Configuration

The application uses the following APIs:
- ECMWF API - [Documentation](https://confluence.ecmwf.int/plugins/servlet/mobile?contentId=47293600#content/view/47293600)
- Meteoblue API - [Documentation](https://docs.meteoblue.com/en/apis/weather-data-packages-api)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.