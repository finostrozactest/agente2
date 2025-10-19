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
st.title("⚡ Asistente de Análisis de Datos")
st.caption("Modo híbrido: rápido o preciso según tu elección.")

@st.cache_data
def load_and_clean_data(url):
    try:
        df = pd.read_excel(url)
        df.dropna(subset=['CustomerID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['CustomerID'] = df['CustomerID'].astype(int)
        df.rename(columns={'InvoiceNo': 'InvoiceID'}, inplace=True)
        return df
    except Exception:
        return None

@st.cache_resource
def get_engine(df):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    df.to_sql("transacciones", engine, index=False, if_exists="replace")
    return engine

@st.cache_resource
def get_llm(streaming=False):
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY en secretos de Streamlit")
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0, streaming=streaming)

@st.cache_resource
def get_agent(engine, llm):
    db = SQLDatabase(engine=engine, include_tables=["transacciones"])
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = """
Eres un analista de datos experto que trabaja con una base de datos SQLite.
Debes planificar, generar y validar tus consultas SQL antes de responder.
"""
    return create_sql_agent(llm=llm, toolkit=toolkit, verbose=False, prefix=prefix, handle_parsing_errors=True, max_iterations=3)

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

def run_direct_query(llm, engine, question):
    try:
        sql_prompt = f"""
Genera una consulta SQL válida en SQLite para responder la siguiente pregunta.
Usa la tabla 'transacciones' y solo devuelve la SQL, sin texto adicional.

Pregunta: {question}
"""
        sql_query = llm.invoke(sql_prompt).content.strip()
        if not sql_query.lower().startswith("select"):
            raise ValueError(f"Consulta no válida generada: {sql_query}")
        df = pd.read_sql_query(sql_query, engine)
        explain_prompt = f"Explica brevemente en español los resultados de esta consulta:\n{question}"
        explanation = llm.invoke(explain_prompt).content
        return explanation, df
    except Exception as e:
        return f"Error al generar o ejecutar la consulta: {e}", None

ecommerce_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
ecommerce_data = load_and_clean_data(ecommerce_url)

if ecommerce_data is None:
    st.error("No se pudieron cargar los datos. Verifica la URL o la conexión.")
    st.stop()

st.sidebar.header("Opciones")
st.sidebar.download_button("📥 Descargar Base de Datos Completa (Excel)", to_excel(ecommerce_data), "datos_completos_ecommerce.xlsx")
modo_agente = st.sidebar.toggle("🧠 Modo agente (más preciso, más lento)", value=False)
use_streaming = st.sidebar.checkbox("Activar streaming", value=False)
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Historial limpiado. ¿En qué te puedo ayudar?"}]
st.sidebar.button("🧹 Limpiar Historial de Chat", on_click=clear_chat_history)

engine = get_engine(ecommerce_data)
llm = get_llm(use_streaming)
agent_executor = get_agent(engine, llm) if modo_agente else None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. ¿Qué deseas analizar?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ej: ¿Cuáles son los 5 productos más vendidos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if modo_agente:
            try:
                if use_streaming:
                    placeholder = st.empty()
                    collected = ""
                    for chunk in agent_executor.stream({"input": prompt}):
                        if "output" in chunk:
                            collected += chunk["output"]
                            placeholder.markdown(collected)
                    final_text = collected
                else:
                    final_text = agent_executor.run(prompt)
            except Exception as e:
                final_text = f"Error al usar el agente: {e}"
            text_part, df_result = parse_response_to_df(str(final_text))
        else:
            explanation, df_result = run_direct_query(llm, engine, prompt)
            text_part = explanation

        st.session_state.messages.append({"role": "assistant", "content": text_part})
        st.markdown(text_part)
        if df_result is not None and not df_result.empty:
            st.dataframe(df_result)
            st.download_button("📊 Descargar Resultado (Excel)", to_excel(df_result), "resultado.xlsx")
