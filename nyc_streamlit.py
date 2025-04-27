# 📦 Imports
import streamlit as st
import requests
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from datetime import datetime

# -------------------------------
# 📊 Define Category Filters
# -------------------------------
CATEGORY_FILTERS = {
    "Parks": '["leisure"="park"]',
    "Shopping Malls": '["shop"="mall"]',
    "Metro Stations": '["railway"="station"]',
    "Nightclubs": '["amenity"="nightclub"]',
    "Restaurants": '["amenity"="restaurant"]',
    "Schools": '["amenity"="school"]',
    "Colleges": '["amenity"="college"]',
    "Universities": '["amenity"="university"]',
    "Bus Stops": '["highway"="bus_stop"]',
    "Train Stations": '["railway"="station"]',
    "Airports": '["aeroway"="aerodrome"]',
    "Museums": '["tourism"="museum"]',
    "Libraries": '["amenity"="library"]',
    "Grocery Stores": '["shop"="supermarket"]'
}

# -------------------------------
# 🧠 Train DNN Model Inside App
# -------------------------------
@st.cache_resource
def train_dnn_model():
    df = pd.read_csv("Fully_cleaned.csv")

    df['bed'] = pd.to_numeric(df['bed'], errors='coerce')
    df['bath'] = df['bath'].replace(r'\+', '', regex=True)
    df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
    df = df.dropna(subset=['bed', 'bath'])

    df['Property Value_log'] = np.log1p(df['Property Value'])

    X = df.drop(columns=['Property Value', 'Property Value_log'])
    y = df['Property Value_log']

    X = pd.get_dummies(X, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping]
    )

    # Predict and calculate metrics
    y_pred_log = model.predict(X_test_scaled).flatten()
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2,
        'Explained Variance': explained_var
    }

    return model, scaler, X.columns.tolist(), metrics

# -------------------------------
# 🗺️ Fetch Amenities
# -------------------------------
def get_nearby_amenities(lat, lon, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query_parts = []
    
    for filter_query in CATEGORY_FILTERS.values():
        query_parts.append(f"""
        node{filter_query}(around:{radius},{lat},{lon});
        way{filter_query}(around:{radius},{lat},{lon});
        relation{filter_query}(around:{radius},{lat},{lon});
        """)
    
    overpass_query = f"""
    [out:json][timeout:25];
    (
        {''.join(query_parts)}
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        
        amenities = []
        for element in data['elements']:
            if 'tags' in element:
                amenity_info = {
                    'name': element['tags'].get('name', 'Unnamed'),
                    'type': element['tags'].get('amenity') or 
                           element['tags'].get('leisure') or 
                           element['tags'].get('shop') or 
                           element['tags'].get('railway') or 
                           element['tags'].get('tourism') or 
                           element['tags'].get('highway') or 
                           'Unknown',
                    'coordinates': (
                        element.get('lat', element.get('center', {}).get('lat')),
                        element.get('lon', element.get('center', {}).get('lon'))
                    )
                }
                if amenity_info['coordinates'][0] is not None:
                    amenities.append(amenity_info)
        
        return amenities
        
    except Exception as e:
        st.error(f"Error fetching amenities: {str(e)}")
        return []

# -------------------------------
# 🌎 Create NYC Map
# -------------------------------
def create_map():
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    m.add_child(folium.LatLngPopup())
    return m

# -------------------------------
# 🚀 Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="NYC Property Price + Amenities", layout="wide")
    st.title("🏙️ NYC Property Price Prediction and Amenities Explorer")

    st.write("Click anywhere on the map to predict property price and see nearby amenities!")

    # Train Model
    with st.spinner('Training DNN model...'):
        model, scaler, feature_columns, metrics = train_dnn_model()

    # Show Metrics
    st.subheader("📈 Model Performance on Test Set")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**RMSE:** ${metrics['RMSE']:,.2f}")
        st.success(f"**MAE:** ${metrics['MAE']:,.2f}")
    with col2:
        st.success(f"**R² Score:** {metrics['R2 Score']:.4f}")
        st.success(f"**Explained Variance:** {metrics['Explained Variance']:.4f}")

    # Show Map
    m = create_map()
    st_data = st_folium(m, width=800, height=500)

    if st_data and st_data["last_clicked"]:
        lat = st_data["last_clicked"]["lat"]
        lon = st_data["last_clicked"]["lng"]

    st.success(f"Clicked at Latitude: {lat:.5f}, Longitude: {lon:.5f}")

    # Optional Inputs
    st.subheader("🏠 Customize Property Details (Optional)")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_bed = st.number_input("Number of Bedrooms", min_value=0, max_value=20, value=0, step=1)
    with col2:
        user_bath = st.number_input("Number of Bathrooms", min_value=0, max_value=20, value=0, step=1)
    with col3:
        user_area = st.number_input("Area (Sqft)", min_value=0, max_value=10000, value=0, step=50)

    # Dropdown for Property Type
    property_types = [
        'Condo for sale', 'Townhouse for sale', 'House for sale',
        'Co-op for sale', 'Pending', 'Coming Soon', 'Contingent',
        'Multi-family home for sale', 'Foreclosure', 'Land for sale',
        'Condop for sale'
    ]

    selected_type = st.selectbox("🏡 Type of Property", property_types)

    # Prepare Input
    input_data = {col: 0 for col in feature_columns}

    # Set numerical features
    if 'Latitude' in input_data:
        input_data['Latitude'] = lat
    if 'Longitude' in input_data:
        input_data['Longitude'] = lon
    if 'bed' in input_data and user_bed > 0:
        input_data['bed'] = user_bed
    if 'bath' in input_data and user_bath > 0:
        input_data['bath'] = user_bath
    if 'Area (Sqft)' in input_data and user_area > 0:
        input_data['Area (Sqft)'] = user_area

    # Set selected Type (one-hot encoded columns)
    for col in feature_columns:
        if col.startswith('Type of House_') and selected_type in col:
            input_data[col] = 1

    # Scale and Predict
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    pred_log_price = model.predict(input_scaled)[0][0]
    predicted_price = np.expm1(pred_log_price)

    st.subheader("💰 Predicted Property Price")
    st.write(f"${predicted_price:,.2f}")

    # Fetch Amenities
    with st.spinner('Fetching nearby amenities...'):
        amenities = get_nearby_amenities(lat, lon, radius=5000)

    if amenities:
        for a in amenities:
            a['distance_km'] = round(geodesic((lat, lon), a['coordinates']).kilometers, 2)

        amenities_df = pd.DataFrame(amenities)
        amenities_df = amenities_df[['name', 'type', 'distance_km']]
        amenities_df.columns = ['Name', 'Type', 'Distance (km)']

        summary_df = amenities_df.groupby('Type').agg(
            Number_of_Amenities=('Name', 'count'),
            Nearest_Distance_km=('Distance (km)', 'min')
        ).reset_index().sort_values('Nearest_Distance_km')

        st.subheader("📋 Amenities Summary")
        st.dataframe(summary_df, use_container_width=True)

        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Amenities Summary as CSV",
            data=csv,
            file_name=f"amenities_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("No amenities found nearby!")

if __name__ == "__main__":
    main()
