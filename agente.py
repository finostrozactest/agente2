import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import os
import re
import io
from io import BytesIO

# Usamos las importaciones de alto nivel recomendadas por LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Asistente de Análisis de Datos",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Asistente de Análisis de Datos con Streaming")
st.caption("Respuestas instantáneas palabra por palabra. Impulsado por Google Gemini y LangChain.")

# --- 1. Carga de Datos y Caché ---
@st.cache_data
def load_and_clean_data(url):
    """Carga y limpia los datos desde una URL."""
    try:
        df = pd.read_excel(url)
        df.dropna(subset=['CustomerID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['CustomerID'] = df['CustomerID'].astype(int)
        df.rename(columns={'InvoiceNo': 'InvoiceID', 'StockCode': 'StockCode', 'Description': 'Description', 'Quantity': 'Quantity', 'InvoiceDate': 'InvoiceDate', 'UnitPrice': 'UnitPrice', 'CustomerID': 'CustomerID', 'Country': 'Country'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# --- 2. Función de Configuración del Agente Único y Mejorado ---
@st.cache_resource
def setup_intelligent_agent(df):
    """
    Crea y cachea una única instancia de un agente inteligente.
    """
    st.info("Inicializando el motor de IA y la base de datos en memoria...")
    
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    df.to_sql("transacciones", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine=engine, include_tables=["transacciones"])
    
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("La clave GOOGLE_API_KEY no se encuentra en los secretos de Streamlit.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = google_api_key
    # Habilitamos el streaming en el modelo
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, streaming=True)

    prefix = """
    Eres un analista de datos experto, meticuloso y auto-crítico que trabaja con una base de datos SQLite.
    Tu objetivo es responder a las preguntas del usuario siguiendo un riguroso proceso de tres pasos en tu pensamiento interno:

    1.  **PENSAMIENTO INICIAL**: Analiza la pregunta del usuario. Identifica qué información se necesita y cómo se puede obtener de la tabla 'transacciones'. Formula un plan y una consulta SQL para ejecutar.
    2.  **AUTO-VALIDACIÓN CRÍTICA**: Antes de dar la respuesta final, detente y critica tu propia consulta SQL. Pregúntate:
        - ¿Esta consulta responde EXACTAMENTE a la pregunta del usuario?
        - ¿Estoy filtrando o agrupando por las columnas correctas?
        - ¿Son los cálculos (ej. ingresos = Quantity * UnitPrice) correctos?
        - ¿Podría la consulta ser más simple o eficiente?
        Si encuentras un error, corrígelo y ejecuta la nueva consulta.

    3.  **RESPUESTA FINAL**: Basado en los resultados de tu consulta validada, proporciona una respuesta clara y concisa en español. Si la respuesta es una tabla, preséntala en formato Markdown.

    La base de datos contiene una única tabla llamada 'transacciones'.
    """
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=prefix,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    st.success("¡Motor de IA listo!")
    return agent_executor

# --- 3. Funciones de Ayuda para la Interfaz (Sin cambios) ---
def to_excel(df_to_convert):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_convert.to_excel(writer, index=False, sheet_name='Resultado')
    return output.getvalue()

# --- Flujo Principal de la Aplicación ---

# --- CORRECCIÓN DE LA URL ---
# El identificador correcto del dataset es 00352.
ecommerce_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
# -----------------------------

ecommerce_data = load_and_clean_data(ecommerce_url)

if ecommerce_data is not None:
    agent_executor = setup_intelligent_agent(ecommerce_data)

    st.sidebar.header("Opciones")
    st.sidebar.download_button(
        label="📥 Descargar Base de Datos Completa (Excel)",
        data=to_excel(ecommerce_data),
        file_name="datos_completos_ecommerce.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet"
    )
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! El historial ha sido limpiado. ¿En qué te puedo ayudar ahora?"}]
    st.sidebar.button("🧹 Limpiar Historial de Chat", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. ¿Qué te gustaría saber?"}]

    # Mostrar historial de chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) # Usamos markdown para renderizar tablas guardadas

    if prompt := st.chat_input("Ej: ¿Cuáles son los 5 productos más vendidos?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- LÓGICA DE STREAMING ---
            response_generator = agent_executor.stream({"input": prompt})
            
            # st.write_stream se encarga de todo el manejo de los fragmentos por nosotros.
            # Solo mostraremos el "output" final que es la respuesta en lenguaje natural.
            def stream_extractor(generator):
                for chunk in generator:
                    if "output" in chunk:
                        yield chunk["output"]
            
            full_response = st.write_stream(stream_extractor(response_generator))

            # Guardamos la respuesta completa en el historial para que las tablas se muestren bien al recargar.
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.error("No se pudieron cargar los datos. Por favor, verifica la conexión a internet o la URL de los datos y refresca la página.")
