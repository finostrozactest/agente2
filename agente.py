import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import os
import re
import io
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

st.set_page_config(page_title="Asistente de Análisis de Datos", page_icon="⚡", layout="wide")
st.title("⚡ Asistente de Análisis de Datos con Streaming")
st.caption("Respuestas palabra por palabra. Impulsado por Google Gemini y LangChain.")

@st.cache_data
def load_and_clean_data(url):
    try:
        df = pd.read_excel(url)
        df.dropna(subset=['CustomerID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['CustomerID'] = df['CustomerID'].astype(int)
        df.rename(columns={'InvoiceNo': 'InvoiceID', 'StockCode': 'StockCode', 'Description': 'Description', 'Quantity': 'Quantity', 'InvoiceDate': 'InvoiceDate', 'UnitPrice': 'UnitPrice', 'CustomerID': 'CustomerID', 'Country': 'Country'}, inplace=True)
        return df
    except Exception as e:
        return None

@st.cache_resource
def setup_intelligent_agent(df, use_streaming: bool):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    df.to_sql("transacciones", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine=engine, include_tables=["transacciones"])
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Falta GOOGLE_API_KEY en secretos de Streamlit")
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0, streaming=use_streaming)
    prefix = """
Eres un analista de datos experto, meticuloso y auto-crítico que trabaja con una base de datos SQLite.
Tu objetivo es responder a las preguntas del usuario siguiendo un riguroso proceso de tres pasos en tu pensamiento interno:

1. PENSAMIENTO INICIAL: Analiza la pregunta del usuario. Identifica qué información se necesita y cómo se puede obtener de la tabla 'transacciones'. Formula un plan y una consulta SQL para ejecutar.
2. AUTO-VALIDACIÓN CRÍTICA: Antes de dar la respuesta final, detente y critica tu propia consulta SQL. Pregúntate si la consulta responde exactamente a la pregunta y si los cálculos son correctos. Corrige si es necesario.
3. RESPUESTA FINAL: Basado en los resultados de tu consulta validada, proporciona una respuesta clara y concisa en español. Si la respuesta es una tabla, preséntala en formato Markdown.
"""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False, prefix=prefix, handle_parsing_errors=True, max_iterations=5)
    return agent_executor

def to_excel(df_to_convert):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_convert.to_excel(writer, index=False, sheet_name='Resultado')
    return output.getvalue()

def parse_response_to_df(response_text: str):
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(response_text)
    if not table_match:
        return response_text, None
    table_str = table_match.group(0)
    text_part = response_text.replace(table_str, "").strip()
    try:
        lines = table_str.strip().split("\n")
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            del lines[1]
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like))
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return text_part if text_part else "Aquí están los resultados de tu consulta:", df
    except Exception:
        return response_text, None

ecommerce_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
ecommerce_data = load_and_clean_data(ecommerce_url)

if ecommerce_data is None:
    st.error("No se pudieron cargar los datos. Verifica la URL o la conexión a Internet.")
else:
    st.sidebar.header("Opciones")
    st.sidebar.download_button(label="📥 Descargar Base de Datos Completa (Excel)", data=to_excel(ecommerce_data), file_name="datos_completos_ecommerce.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet")
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! El historial ha sido limpiado. ¿En qué te puedo ayudar ahora?"}]
    st.sidebar.button("🧹 Limpiar Historial de Chat", on_click=clear_chat_history)
    use_streaming = st.sidebar.checkbox("Habilitar streaming (más detallado pero puede ser más lento)", value=False)
    try:
        agent_executor = setup_intelligent_agent(ecommerce_data, use_streaming)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. ¿Qué te gustaría saber?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: ¿Cuáles son los 5 productos más vendidos?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if use_streaming:
                placeholder = st.empty()
                collected = ""
                try:
                    for chunk in agent_executor.stream({"input": prompt}):
                        if "output" in chunk:
                            collected += chunk["output"]
                            placeholder.markdown(collected)
                    final_text = collected
                except Exception as e:
                    final_text = f"Error durante el streaming: {e}"
                    placeholder.markdown(final_text)
            else:
                try:
                    final_text = agent_executor.run(prompt)
                except Exception as e:
                    final_text = f"Error al ejecutar el agente: {e}"

            text_part, df_result = parse_response_to_df(final_text if isinstance(final_text, str) else str(final_text))
            st.session_state.messages.append({"role": "assistant", "content": text_part})
            with st.chat_message("assistant"):
                st.markdown(text_part)
                if df_result is not None:
                    st.data_editor(df_result, disabled=True)
                    st.download_button(label="📥 Descargar Resultado (Excel)", data=to_excel(df_result), file_name="resultado_consulta.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet")
