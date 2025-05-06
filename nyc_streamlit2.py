import streamlit as st
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from xgboost import XGBRegressor
from datetime import datetime

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

@st.cache_resource
def train_dnn_model():
    df = pd.read_csv("Fully_cleaned.csv")

    zipcode_safety_mapping = df[['Zip Code', 'Safety_Score']].dropna().drop_duplicates()
    zipcode_to_safety = dict(zip(zipcode_safety_mapping['Zip Code'], zipcode_safety_mapping['Safety_Score']))

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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train_scaled,
        y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping]
    )

    y_pred_log = model.predict(X_test_scaled).flatten()
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }

    return model, scaler, X.columns.tolist(), metrics, zipcode_to_safety

@st.cache_resource
def train_xgb_model():
    df = pd.read_csv("Fully_cleaned.csv")

    target = 'Property Value'
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols.remove(target)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    X = df[categorical_cols + numerical_cols]
    y = df[target]
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.1,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='rmse'
    )

    model.fit(X_train_transformed, y_train_log, eval_set=[(X_train_transformed, y_train_log), (X_test_transformed, y_test_log)], verbose=0)

    y_pred_log = model.predict(X_test_transformed)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

    return model, preprocessor, categorical_cols, numerical_cols, metrics

def create_map():
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    m.add_child(folium.LatLngPopup())
    return m

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

def main():
    st.set_page_config(page_title="NYC Property Price + Amenities", layout="wide")
    st.title("🏙️ NYC Property Price Prediction + Amenities Explorer")

    st.write("Click anywhere on the map to predict property price using DNN and XGBoost models, and explore nearby amenities!")

    # Load Models
    with st.spinner('Training DNN Model...'):
        dnn_model, dnn_scaler, dnn_feature_cols, dnn_metrics, zipcode_to_safety = train_dnn_model()

    with st.spinner('Training XGBoost Model...'):
        xgb_model, xgb_preprocessor, xgb_cat_cols, xgb_num_cols, xgb_metrics = train_xgb_model()

    # Show Model Metrics
    st.subheader("📈 Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("DNN RMSE", f"${dnn_metrics['RMSE']:,.2f}")
        st.metric("DNN MAE", f"${dnn_metrics['MAE']:,.2f}")
        st.metric("DNN R² Score", f"{dnn_metrics['R2 Score']:.4f}")
    with col2:
        st.metric("XGBoost RMSE", f"${xgb_metrics['RMSE']:,.2f}")
        st.metric("XGBoost MAE", f"${xgb_metrics['MAE']:,.2f}")
        st.metric("XGBoost R² Score", f"{xgb_metrics['R2 Score']:.4f}")

    # Show Map
    m = create_map()
    st_data = st_folium(m, width=800, height=500)

    if st_data and st_data["last_clicked"]:
        lat = st_data["last_clicked"]["lat"]
        lon = st_data["last_clicked"]["lng"]

        st.success(f"Clicked at Latitude: {lat:.5f}, Longitude: {lon:.5f}")

        # User Inputs
        st.subheader("🏠 Property Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            user_bed = st.number_input("Bedrooms", min_value=0, max_value=20, value=0, step=1)
        with col2:
            user_bath = st.number_input("Bathrooms", min_value=0, max_value=20, value=0, step=1)
        with col3:
            user_area = st.number_input("Area (Sqft)", min_value=0, max_value=10000, value=0, step=50)

        property_types = [
            'Condo for sale', 'Townhouse for sale', 'House for sale',
            'Co-op for sale', 'Pending', 'Coming Soon', 'Contingent',
            'Multi-family home for sale', 'Foreclosure', 'Land for sale', 'Condop for sale'
        ]
        selected_type = st.selectbox("🏡 Property Type", property_types)

        # Get Zip Code
        geolocator = Nominatim(user_agent="nyc-property-app")
        location = geolocator.reverse((lat, lon), exactly_one=True)
        zipcode = None
        if location and 'postcode' in location.raw['address']:
            zipcode = location.raw['address']['postcode']
            st.info(f"Detected Zip Code: {zipcode}")

        # Fetch Amenities
        with st.spinner('Fetching nearby amenities...'):
            amenities = get_nearby_amenities(lat, lon, radius=5000)

        amenities_count = {}
        nearest_distance = {}

        if amenities:
            amenities_df = pd.DataFrame(amenities)
            amenities_df['Distance (km)'] = amenities_df['coordinates'].apply(lambda coord: geodesic((lat, lon), coord).kilometers)

            amenities_df['Type'] = amenities_df['type'].str.lower()
            for feature in CATEGORY_FILTERS.keys():
                feature_lower = feature.lower()
                count = amenities_df['Type'].str.contains(feature_lower.split()[0]).sum()
                min_distance = amenities_df[amenities_df['Type'].str.contains(feature_lower.split()[0])]['Distance (km)'].min()

                amenities_count[f"Number_of_{feature.replace(' ', '_')}_Nearby"] = count
                nearest_distance[f"Distance_to_Nearest_{feature.replace(' ', '_')}_miles_binned"] = min_distance * 0.621371 if not pd.isna(min_distance) else 10.0

        # Prepare Input for DNN
        input_data = {col: 0 for col in dnn_feature_cols}
        if 'Latitude' in input_data:
            input_data['Latitude'] = lat
        if 'Longitude' in input_data:
            input_data['Longitude'] = lon
        if 'bed' in input_data:
            input_data['bed'] = user_bed
        if 'bath' in input_data:
            input_data['bath'] = user_bath
        if 'Area (Sqft)' in input_data:
            input_data['Area (Sqft)'] = user_area
        if 'Safety_Score' in input_data:
            input_data['Safety_Score'] = zipcode_to_safety.get(int(zipcode), np.median(list(zipcode_to_safety.values()))) if zipcode else np.median(list(zipcode_to_safety.values()))
        for col in dnn_feature_cols:
            if col.startswith('Type of House_') and selected_type in col:
                input_data[col] = 1
        for col in dnn_feature_cols:
            if col in amenities_count:
                input_data[col] = amenities_count[col]
            if col in nearest_distance:
                input_data[col] = nearest_distance[col]

        input_df = pd.DataFrame([input_data])
        input_scaled = dnn_scaler.transform(input_df)

        dnn_pred_log = dnn_model.predict(input_scaled)[0][0]
        dnn_predicted_price = np.expm1(dnn_pred_log)

        # Prepare Input for XGBoost
        xgb_input_dict = {}

                # Categorical features
        for col in xgb_cat_cols:
            if col == 'Type of House':
                xgb_input_dict[col] = selected_type
            elif col == 'borough' and 'borough' in input_data:
                xgb_input_dict[col] = input_data['borough']
            elif col == 'Zip Code' and zipcode:
                xgb_input_dict[col] = int(zipcode)
            else:
                xgb_input_dict[col] = "Unknown"

        for col in xgb_num_cols:
            if col in input_data:
                xgb_input_dict[col] = input_data[col]
            else:
                xgb_input_dict[col] = 0

        xgb_input_df = pd.DataFrame([xgb_input_dict])
        xgb_input_df = xgb_input_df.reindex(columns=xgb_cat_cols + xgb_num_cols, fill_value=0)
        xgb_input_transformed = xgb_preprocessor.transform(xgb_input_df)

        xgb_pred_log = xgb_model.predict(xgb_input_transformed)[0]
        xgb_predicted_price = np.expm1(xgb_pred_log)

        # Display Predictions
        st.subheader("💰 Predicted Property Prices")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"💻 DNN Prediction: ${dnn_predicted_price:,.2f}")
        with col2:
            st.info(f"🌲 XGBoost Prediction: ${xgb_predicted_price:,.2f}")

if __name__ == "__main__":
    main()
