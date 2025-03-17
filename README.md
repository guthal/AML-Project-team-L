# AML-Project-team-L

# Nearby Amenities Finder

A Streamlit-based web application that helps users find various amenities (schools, parks, restaurants, etc.) near a specified location using OpenStreetMap data.

## 🎯 Features

- 🔍 Search for 14 different types of amenities:
  - Parks
  - Shopping Malls
  - Metro Stations
  - Nightclubs
  - Restaurants
  - Schools
  - Colleges
  - Universities
  - Bus Stops
  - Train Stations
  - Airports
  - Museums
  - Libraries
  - Grocery Stores
- 🗺️ Interactive map visualization
- 📊 Tabular data view
- 📈 Summary statistics
- 📥 CSV export functionality
- ⏱️ Processing time tracking

## 🛠️ Technology Stack

- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **OpenStreetMap**: Data source via Overpass API
- **Folium**: Map visualization
- **Pandas**: Data manipulation and analysis

## 📦 Required Packages

bash
pip install streamlit
pip install streamlit-folium
pip install folium
pip install pandas
pip install requests
pip install geopy


## 💻 Usage

1. Enter latitude and longitude coordinates
2. Select an amenity category from the dropdown
3. Click "Search Amenities"
4. View results in three different formats:
   - Interactive map view
   - Detailed table view
   - Summary statistics

## 🔍 API Details

The application uses the Overpass API to query OpenStreetMap data:
- Default search radius: 5 kilometers
- Timeout: 25 seconds
- Supports nodes, ways, and relations

## ⚙️ Performance Considerations

- Results are cached to prevent unnecessary API calls
- Session state maintains data between interactions
- Automatic retries for failed API requests
- Progress indicators for long-running operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- TeamL
- Advanced Machine Learning Course Project

## 🙏 Acknowledgments

- OpenStreetMap contributors
- Streamlit community
- Folium developers

# Instructions to Run Nearby Amenities Finder

## Prerequisites

1. **Python Installation**
   - Download and install Python 3.7 or higher from [python.org](https://python.org)
   - Ensure Python is added to your system PATH

2. **Git Installation** (optional)
   - Download and install Git from [git-scm.com](https://git-scm.com)

## Step-by-Step Setup

### 1. Get the Code

**Option A: Using Git**

bash
Clone the repository
git clone https://github.com/your-username/AML-Project-team-L.git
cd AML-Project-team-L


**Option B: Direct Download**
- Download the ZIP file from GitHub
- Extract to your preferred location
- Open terminal/command prompt in the extracted folder

### 2. Set Up Virtual Environment

**Windows**

bash
Create virtual environment
python -m venv venv
Activate virtual environment
venv\Scripts\activate


**MacOS/Linux**

bash
Create virtual environment
python3 -m venv venv
Activate virtual environment
source venv/bin/activate

### 3. Install Required Packages

bash
Install all required packages
pip install streamlit streamlit-folium folium pandas requests geopy


Or using requirements.txt:

bash
pip install -r requirements.txt


### 4. Run the Application

bash
Run the Streamlit app
streamlit run nearby_schools_streamlit.py


The application should automatically open in your default web browser.
If not, copy the URL shown in the terminal (usually http://localhost:8501)

## Using the Application

1. **Input Location**
   - Enter latitude (e.g., 51.5074 for London)
   - Enter longitude (e.g., -0.1278 for London)

2. **Select Amenity**
   - Choose from the dropdown menu:
     - Parks
     - Shopping Malls
     - Restaurants
     - Schools
     - etc.

3. **Search and View Results**
   - Click "Search Amenities"
   - View results in three tabs:
     - Map View: Interactive map with markers
     - Table View: Detailed list of amenities
     - Summary View: Statistics and overview

4. **Export Data**
   - Use the "Download Data as CSV" button to save results

## Troubleshooting

### Common Issues and Solutions

1. **"streamlit: command not found"**
   ```bash
   # Reinstall streamlit
   pip uninstall streamlit
   pip install streamlit
   ```

2. **Port Already in Use**
   ```bash
   # Kill the process using the port
   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F

   # MacOS/Linux
   lsof -i :8501
   kill -9 <PID>
   ```

3. **Package Installation Errors**
   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install packages individually
   pip install streamlit
   pip install streamlit-folium
   pip install folium
   pip install pandas
   pip install requests
   pip install geopy
   ```

4. **Virtual Environment Issues**
   ```bash
   # Deactivate and recreate
   deactivate
   rm -rf venv
   python -m venv venv
   # Activate again using step 2
   ```

## Additional Tips

1. **For Better Performance**
   - Close other applications
   - Use a stable internet connection
   - Start with small search radius

2. **Data Export**
   - Results are automatically cached
   - Download CSV before starting new search
   - Clear results using the button provided

3. **Map Navigation**
   - Zoom: Mouse wheel or +/- buttons
   - Pan: Click and drag
   - Reset view: Double click

4. **Best Practices**
   - Start with known coordinates
   - Test different amenity types
   - Allow time for API responses
   - Clear results between searches

## Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify your Python version: `python --version`
3. Confirm all packages are installed: `pip list`
4. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information

## Updates and Maintenance

To update the application:

bash
Pull latest changes (if using Git)
git pull origin main
Update packages
pip install -r requirements.txt --upgrade


## Development Mode

For developers who want to modify the code:

bash
Install development dependencies
pip install black flake8 pytest
Run tests
pytest tests/
Format code
black .

```

This instructions file provides:
1. Complete setup instructions
2. Usage guidelines
3. Troubleshooting steps
4. Best practices
5. Maintenance procedures
6. Development setup

## 📞 Support

For support or questions, please open an issue in the GitHub repository.