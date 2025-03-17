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

4. Run the tests to verify functionality:
```bash
python -m unittest discover tests
```

5. Run the application:
```bash
python -m app.app --debug
```

6. Open your web browser and navigate to:
```
http://localhost:5000
```

### Deployment on Render.com

1. Fork/push this repository to your GitHub account.

2. Sign up for a [Render account](https://render.com/).

3. Create a new Web Service on Render:
   - Connect your GitHub repository
   - Select "Python" as the environment
   - Use `./build.sh` as the build command
   - Use `gunicorn 'app.app:create_app()' --bind=0.0.0.0:$PORT --workers=2` as the start command
   - Add any necessary environment variables

4. Deploy:
   - Render will automatically build and deploy your application
   - Access your app at the URL provided by Render

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