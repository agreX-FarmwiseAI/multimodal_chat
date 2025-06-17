import streamlit as st
import geopandas as gpd
import pandas as pd
import psycopg2
import os
import json
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import VARCHAR
from geoalchemy2 import Geometry
from streamlit_folium import st_folium
import folium
import litellm 
from litellm import completion
from io import BytesIO
import tempfile
from shapely import wkt
from typing import Dict, Any, List, Optional
from shapely import wkb, wkt
from shapely.geometry import mapping
from shapely.geometry import shape
import geojson
from geoalchemy2 import WKTElement
import binascii
# ---------- CONFIGURATION ---------- #
st.set_page_config(layout="wide")
DATABASE_URL = "postgresql://postgres:j5<lOZgOfN6im:~VCoTao6**NZSP@fai-apps-pg-db.cdyc2k8eytnr.ap-south-1.rds.amazonaws.com:5432/MapsAI"
engine = create_engine(DATABASE_URL)
litellm.drop_params = True
litellm.telemetry = False
litellm.api_key = "AIzaSyACE6JsSmb9jMyhf8q-ZpPcbt8Xid1cVak"
model_name = "gemini/gemini-2.5-flash-preview-04-17"
database_schema = ""
DYNAMIC_PROMPT_TEMPLATE = """
You are an expert PostgreSQL/PostGIS query writer. Your primary function is to convert a user's natural language question into a single, precise, and executable SQL query based on the provided database schema, capabilities, and a strict set of rules.

### DATABASE CONTEXT
{schema_definition}

### GEOSPATIAL CAPABILITIES
- The database has PostGIS extensions enabled.
- Key Functions: `ST_Intersects`, `ST_DWithin`, `ST_Buffer`, `ST_Area`, `ST_Length`, `ST_Centroid`, `ST_Union`, `ST_Within`.
- Pay close attention to the geometry column names provided in the schema (e.g., `geometry`, `geometry_wkt`).
- Assume all geometries are in a compatible coordinate system for direct comparison.

---

### CRITICAL RULES FOR QUERY GENERATION
1.  **The Geometry Rule (MANDATORY):**
    *   IF the user's question contains any of these keywords: `show`, `display`, `view`, `plot`, `render`, `draw`, `locate`.
    *   THEN the `SELECT` statement MUST include the relevant geometry column as the very first column in the result.
    *   ELSE (if none of those keywords are present), the query MUST NOT include any geometry columns in the final `SELECT` list.
2.  **Geospatial Operations:**
    *   Proximity ("near"): For "features near a location", use a spatial join with `ST_Intersects` or `ST_DWithin`.
    *   Buffering ("buffer", "expand"): Generate a new geometry using `ST_Buffer(geometry, distance)`.
    *   Relationships ("between", "connecting"): For features connecting two other features, find the geometries of the two anchor features and then find the third feature that `ST_Intersects` with BOTH anchor geometries.
3.  **Attribute & Logic:**
    *   Ranking/Aggregation ("most", "least", "rank"): Use `GROUP BY` with aggregate functions (`COUNT`, `SUM`, `AVG`) and window functions (`RANK()`).
    *   Progress/Completion ("progress of", "completion percentage"): If the schema contains columns that represent stages (like `_3_1`, `_3_2`), interpret progress questions as calculating the percentage of rows where that column is `NOT NULL`.
    *   Filtering: Use `ILIKE` for case-insensitive text matching on names.
4.  **Output Format:**
    *   Your final output must be ONLY the SQL query.
    *   Do not include explanations, comments, or markdown formatting like ```sql.

---
### USER QUESTION TO PROCESS:
{user_question}
"""

# Initialize session state
if 'uploaded_tables' not in st.session_state:
    st.session_state.uploaded_tables = []
    
if 'current_response' not in st.session_state:
    st.session_state.current_response = {"user": "", "ai": ""}
    
if 'geojson_data' not in st.session_state:
    st.session_state.geojson_data = None

# ---------- FUNCTIONS ---------- #
def convert_geometries_to_geojson(geometries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert a list of geometry dicts (with 'geometry' key as WKB hex, WKT, GeoJSON string, or GeoJSON dict)
    to a GeoJSON FeatureCollection.
    Handles various geometry types.
    All other columns are included in the properties of each feature.
    """
    features = []
    for geom_obj in geometries:
        # Find the key that contains "geometry" or "buffer" (exact or substring match)
        geom_key = next(
            (k for k in geom_obj.keys() if "geometry" in k.lower() or "buffer" in k.lower()),
            None
        )
        geom = geom_obj.get(geom_key) if geom_key else None
        shapely_geom = None

        if isinstance(geom, dict):
            try:
                # Handle GeoJSON dict
                shapely_geom = shape(geom)
            except Exception as e:
                print(f"Could not parse GeoJSON dict: {e}")
                shapely_geom = None
        elif isinstance(geom, str):
            try:
                # Try WKB hex
                shapely_geom = wkb.loads(binascii.unhexlify(geom))
            except Exception:
                try:
                    # Try WKT
                    shapely_geom = wkt.loads(geom)
                except Exception:
                    try:
                        # Try loading as GeoJSON string
                        shapely_geom = shape(json.loads(geom))
                    except Exception as e:
                        print(f"Could not parse geometry string: {e}")
                        shapely_geom = None
        elif isinstance(geom, (bytes, bytearray)):
            try:
                # Handle WKB bytes
                shapely_geom = wkb.loads(geom)
            except Exception as e:
                print(f"Could not parse WKB bytes: {e}")
                shapely_geom = None

        if shapely_geom is not None and shapely_geom.is_valid:
            # All other columns except 'geometry' go into properties
            properties = {k: v for k, v in geom_obj.items() if k != geom_key}
            feature = geojson.Feature(geometry=shapely_geom.__geo_interface__, properties=properties)
            features.append(feature)
        elif shapely_geom is not None and not shapely_geom.is_valid:
            print(f"Invalid geometry skipped: {geom}")

    return geojson.FeatureCollection(features)

def frame_response_from_sql_results(result):
    print(f"sql_results: {result}")

    #if isinstance(result, list) and all(isinstance(row, tuple) for row in result):
    print("YEs")
    geojson_features = []
    result_without_geometry = []

    for row in result:
        if not row:
            continue

        geometry_candidate = row[0]
        try:
            # Try parsing as WKB hex
            if isinstance(geometry_candidate, str) and geometry_candidate.startswith('010'):  # basic check
                geometry = wkb.loads(bytes.fromhex(geometry_candidate))
            elif isinstance(geometry_candidate, bytes):
                geometry = wkb.loads(geometry_candidate)
            else:
                continue

            # Build GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {}
            }

            # Handle remaining data as properties
            if len(row) > 1:
                if isinstance(row[1], dict):
                    feature["properties"] = row[1]
                    result_without_geometry.append(row[1])
                else:
                    props = {f"col_{i}": val for i, val in enumerate(row[1:], start=1)}
                    feature["properties"] = props
                    result_without_geometry.append(props)
            else:
                result_without_geometry.append({})

            geojson_features.append(feature)

        except Exception as e:
            print(f"Error processing row as geometry: {e}")

    if geojson_features:
        return {
            "result": result_without_geometry if len(result_without_geometry) > 1 else result_without_geometry[0],
            "geojson": {
                "type": "FeatureCollection",
                "features": geojson_features
            }
        }

    # fallback
    return {"result": result, "geojson": None}

def chat_llm(user_input):
    # Simplified for demo - in real implementation, this would return geojson too
    
    sql_results = ""
    sql_query = generate_sql_query_dynamic(user_input)
    print(f"sql_query:{sql_query}")
    with engine.begin() as conn:
        sql_results = conn.execute(text(sql_query)).fetchall()

    result = frame_response_from_sql_results(sql_results)
    result['user_query'] = user_input       
    # Generate final response
    final_response = generate_final_response(result)
    result["final_response"] = final_response
           
    return result["final_response"],result["geojson"]

def generate_final_response(results: Dict[str, Any]) -> str:
        """Generate a cohesive final response from sub-agent results."""
        try:
            #print("results", results['result'])
            #print("user_query", results['user_query'])
            response_prompt = f"""

                    User Query: {results['user_query']}          
                    Result:
                    {results['result']}
                    
                    Generate a user-friendly response that:
                    1. Directly answers the user's question
                    2. Synthesizes information from the result
                    3. Maintains a helpful, professional tone
                    """
           
            msg = [{"role": "user", "content": f"{response_prompt}"}]
            response = completion(model=model_name, messages=msg)
            return response.choices[0].message.content.strip()
           
        except Exception as e:
            print(f"Error generating final response: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
 
def call_llm(prompt):
    msg = [{"role": "user", "content": f"{prompt}"}]
    response = completion(model=model_name, messages=msg)
    return response.choices[0].message.content.strip()

def get_dynamic_schema_from_db(schema_name: str = 'public') -> dict:   
    schema_data = {}
    
    with engine.begin() as conn:
        # Get all table names
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema_name AND table_type = 'BASE TABLE';
        """), {"schema_name": schema_name})
        
        tables = result.fetchall()

        for table_tuple in tables:
            table_name = table_tuple[0]
            full_table_name = f'{schema_name}."{table_name}"'
            schema_data[full_table_name] = {'columns': {}, 'description': ''}

            # Get table description
            desc_result = conn.execute(text("""
                SELECT obj_description(c.oid, 'pg_class')
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relkind = 'r' AND n.nspname = :schema_name AND c.relname = :table_name;
            """), {"schema_name": schema_name, "table_name": table_name}).fetchone()

            if desc_result and desc_result[0]:
                schema_data[full_table_name]['description'] = desc_result[0]

            # Get columns
            cols_result = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = :schema_name AND table_name = :table_name;
            """), {"schema_name": schema_name, "table_name": table_name})

            columns = cols_result.fetchall()
            for col_tuple in columns:
                col_name = col_tuple[0]
                col_type = col_tuple[1]
                schema_data[full_table_name]['columns'][col_name] = col_type
                
    return schema_data


def format_schema_for_prompt():
    schema_data = get_dynamic_schema_from_db("public")
    schema_string = ""
    for table_name, table_info in schema_data.items():
        schema_string += f'TABLE: {table_name}\n'
        if table_info.get('description'):
            schema_string += f"DESCRIPTION: {table_info['description']}\n"
        
        columns_str = ", ".join([f"{col_name} ({col_type})" for col_name, col_type in table_info['columns'].items()])
        schema_string += f"COLUMNS: {columns_str}\n\n"
        
    return schema_string.strip()

def generate_sql_query_dynamic(user_question: str) -> str:
    """
    Generates a SQL query using a dynamic schema.

    Args:
        user_question: The natural language question from the user.
        schema_definition: The formatted string describing the database schema.

    Returns:
        A string containing the generated SQL query.
    """
    database_schema = st.session_state.schema_summary
    final_prompt = DYNAMIC_PROMPT_TEMPLATE.format(
        schema_definition=database_schema,
        user_question=user_question
    )
    print(f"Database Schema : {database_schema}")
    messages = [{"role": "user", "content": final_prompt}]

    try:
        print("Sending request to Gemini with dynamic schema...")
        response = completion(
            model="gemini/gemini-1.5-pro-latest",
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        print("Response received.")
        sql_query = response.choices[0].message.content.strip()

        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        return sql_query.strip()
    except Exception as e:
        return f"An error occurred while calling the API: {e}"


from sqlalchemy import text
import streamlit as st

def delete_all_tables():
    with engine.begin() as conn:
        inspector = inspect(conn)
        table_names = [
            table for table in inspector.get_table_names(schema='public')
            if table not in (
                'geometry_columns',
                'geography_columns',
                'spatial_ref_sys',
                'raster_columns',
                'raster_overviews',
                'topology'
            )
        ]

        for table in table_names:
            conn.execute(text(f'DROP TABLE IF EXISTS public."{table}" CASCADE'))
        conn.commit()

    # Reset session state
    st.session_state.uploaded_tables = []
    st.session_state.current_response = {"user": "", "ai": ""}
    st.session_state.geojson_data = None
    st.success("All uploaded tables have been deleted from the schema.")


def save_to_postgres(df, table_name):
    geom_col = None
    for col in df.columns:
        if df[col].astype(str).str.startswith(('POINT', 'LINESTRING', 'POLYGON', 'MULTIPOLYGON', 'MULTILINESTRING')).any():
            geom_col = col
            break

    if geom_col:
        df = df.rename(columns={geom_col: 'geometry'})      
        # Convert WKT string to WKTElement with SRID 4326
        #df['geometry'] = df['geometry'].apply(wkt.loads)
        df['geometry'] = df['geometry'].apply(lambda x: WKTElement(x.wkt if hasattr(x, "wkt") else x, srid=4326))
        # Convert to GeoDataFrame for safety
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        # Save to PostGIS
        gdf.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            dtype={'geometry': Geometry('GEOMETRY', srid=4326)}
        )
    else:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            dtype={col: VARCHAR for col in df.columns}
        )

def read_file(uploaded_file):
    suffix = uploaded_file.name.split('.')[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        if suffix == 'shp':
            return gpd.read_file(file_path)
        elif suffix == 'geojson':
            return gpd.read_file(file_path)
        elif suffix == 'csv':
            return pd.read_csv(file_path)
        elif suffix in ['xls', 'xlsx']:
            return pd.read_excel(file_path)
        elif suffix == 'gpkg':
            return gpd.read_file(file_path, layer=0)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None

def display_map(geojson_data):
    if not geojson_data:
        # Create base map with satellite layer
        m = folium.Map(location=[11.0, 78.0], zoom_start=6)

        # Add satellite layer
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)

        folium.LayerControl().add_to(m)
        return st_folium(m, width=900, height=600)

    # Convert geojson_data to GeoDataFrame to get bounds
    if isinstance(geojson_data, dict):
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    else:
        gdf = gpd.GeoDataFrame.from_features(json.loads(geojson_data)["features"])

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Create map with satellite as default, centered initially (will update with fit_bounds)
    m = folium.Map(zoom_start=6)

    # Add satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Add the GeoJSON data
    folium.GeoJson(geojson_data, name="GeoData").add_to(m)

    # Fit map to GeoJSON bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    folium.LayerControl().add_to(m)
    return st_folium(m, width=900, height=600)

def process_uploaded_files(uploaded_files):
    global database_schema
    uploaded_tables = []
    with st.spinner("Processing files..."):
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / total_files, text=f"Processing {uploaded_file.name}...")
            df = read_file(uploaded_file)
            if df is not None:
                table_name = uploaded_file.name.replace('.', '_').lower()
                save_to_postgres(df, table_name)
                uploaded_tables.append(table_name)
        
        progress_bar.progress(1.0, text="Generating schema metadata...")
        database_schema = format_schema_for_prompt()
        st.session_state.schema_summary = database_schema
        
    return uploaded_tables

# ---------- UI LAYOUT ---------- #
st.title("🗺️ AI-Powered Geospatial Chat Application")
st.markdown("""
Upload your geospatial data and chat with it using an intelligent AI.<br>
Supported formats: `.shp`, `.geojson`, `.csv`, `.xlsx`, `.gpkg`
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Data Management")
    uploaded_files = st.file_uploader("Upload geospatial files", type=["shp", "geojson", "csv", "xlsx", "gpkg"], accept_multiple_files=True)

    if uploaded_files:
        st.success("Files uploaded. Confirm to convert to database.")
        if st.button("Upload"):
            st.session_state.uploaded_tables = process_uploaded_files(uploaded_files)
            
            st.success("All files processed and schema metadata generated!")
            # Initialize AI greeting
            st.session_state.current_response = {"user": "", "ai": "Your geospatial data is ready! Ask me anything about it."}
    
    if st.button("❌ Delete Exists"):
        delete_all_tables()

# Create persistent columns layout
col1, col2 = st.columns([1, 1], gap="medium")

# ---------- LEFT COLUMN: CHAT INTERFACE ---------- #
with col1:
    st.subheader("💬 Chat with Your Data")
    
    # Container for chat messages with scrollable content
    chat_container = st.container(height=400, border=True)
    
    if st.session_state.uploaded_tables:
        # Display only the latest exchange
        if st.session_state.current_response["user"]:
            with chat_container:
                with st.chat_message("user"):
                    st.write(st.session_state.current_response["user"])
        
        if st.session_state.current_response["ai"]:
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(st.session_state.current_response["ai"])
    
    # Chat input at the bottom of the left column
    if prompt := st.chat_input("Ask a question about your data...", key="chat_input"):
        if not st.session_state.uploaded_tables:
            st.warning("Please upload data first")
            st.stop()
            
        # Update current response (only showing latest exchange)
        st.session_state.current_response["user"] = prompt
        
        # Get AI response
        with st.spinner("Thinking..."):
            text_response, geojson_output = chat_llm(prompt)
            print(f"text:{text_response},Geojson:{geojson_output}")
            st.session_state.current_response["ai"] = text_response
            st.session_state.geojson_data = geojson_output
            st.rerun()

# ---------- RIGHT COLUMN: MAP VIEW ---------- #
with col2:
    st.subheader("🗺️ Map View")
    
    # Container for map with fixed height
    map_container = st.container(height=600, border=False)
    
    if st.session_state.uploaded_tables:
        with map_container:
            display_map(st.session_state.geojson_data)
    else:
        with map_container:
            # Show empty satellite map
            m = folium.Map(location=[11.0, 78.0], zoom_start=6)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            # folium.TileLayer(
            #     tiles='openstreetmap',
            #     attr='OpenStreetMap',
            #     name='OpenStreetMap',
            #     overlay=False,
            #     control=True
            # ).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=900, height=550)

# ---------- STYLE ---------- #
st.markdown("""
<style>
    /* Main layout */
    .stApp > div:first-child {
        padding-top: 1rem;
    }
    
    /* Columns layout */
    [data-testid="column"] {
        padding: 0 1rem;
    }
    
    /* Chat container */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:first-child {
        gap: 0.5rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        max-width: 90%;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
        margin-left: auto;
    }
    .stChatMessage.assistant {
        background-color: #e6f7ff;
        margin-right: auto;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
        margin-bottom: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Map container */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:last-child {
        min-height: 600px;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background: white;
        z-index: 100;
        padding-top: 1rem;
        padding-bottom: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
