import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Configuración Global de Gráficos Plotly
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.xaxis.showgrid = False
pio.templates["plotly_white"].layout.xaxis.gridcolor = "lightgray"
pio.templates["plotly_white"].layout.xaxis.gridwidth = 1
pio.templates["plotly_white"].layout.xaxis.fixedrange = True  # Desactiva zoom/pan horizontal
pio.templates["plotly_white"].layout.yaxis.showgrid = False
pio.templates["plotly_white"].layout.yaxis.gridcolor = "lightgray"
pio.templates["plotly_white"].layout.yaxis.gridwidth = 1
pio.templates["plotly_white"].layout.yaxis.fixedrange = True  # Desactiva zoom/pan vertical
pio.templates["plotly_white"].layout.font.color = "black"
# Optimización Móvil Global (Márgenes ajustados al límite para maximizar espacio táctil)
pio.templates["plotly_white"].layout.margin = dict(l=0, r=0, t=60, b=0)
pio.templates["plotly_white"].layout.legend = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)

st.set_page_config(page_title="Observatorio de Sesgo", layout="wide")

st.markdown("## Estudio Cuantitativo de Agenda-Setting: Aplicación de Procesamiento de Lenguaje Natural del Encuadre Comunicativo en la Prensa Colombiana en época electoral (Enero - Junio 2026)")

with st.expander("METODOLOGÍA DEL ESTUDIO", expanded=False):
    st.markdown("""
    La presente plataforma web expone los resultados de un análisis computacional aplicado a la cobertura política de los medios de comunicación en Colombia, correspondiente al periodo de enero a principios de junio de 2026. El objetivo es cuantificar las dinámicas de Agenda-Setting y perfilamiento editorial mediante técnicas de Ciencia de Datos y Procesamiento de Lenguaje Natural (NLP). El estudio se desarrolló a través del siguiente proceso metodológico:

    1. **Extracción y Estructuración:** Recopilación mediante web scraping de un corpus representativo, recolectando entre una y dos noticias diarias de los seis medios masivos principales de Colombia.

    2. **Preprocesamiento de Texto:** Purga de ruido estructural (avisos legales, menús) y limpieza lingüística para aislar el contenido periodístico de valor.

    3. **Clasificación mediante Deep Learning:** Aplicación de modelos Transformers (RoBERTa / DeBERTa) para cuantificar la probabilidad de sentimiento negativo y categorizar el encuadre discursivo.

    4. **Modelado Estadístico:** Agrupación en series de tiempo y aplicación de modelos predictivos para identificar correlaciones causales.
    """)

@st.cache_data
def load_data():
    import os
    DATA_PATH = "data/03_processed/corpus_nlp_features_completo.parquet"
    
    # Fallback por si el archivo se subió a la raíz del repositorio en GitHub
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "corpus_nlp_features_completo.parquet"
        
    # Cache invalidado: cargando data nueva post-pipeline (Ensemble Zero-Shot + SpaCy Logic Restored)
    df = pd.read_parquet(DATA_PATH)
    
    if 'medio' in df.columns:
        df['medio_emisor'] = df['medio'].str.upper()
    else:
        df['medio'] = df['url'].str.extract(r'(?:https?://)?(?:www\.)?([^/.]+)\.')
        df['medio_emisor'] = df['medio'].str.upper()
        
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'], errors='coerce')
    df['fecha_dia'] = df['fecha_publicacion'].dt.date
    return df

df = load_data()

# Se quita el filtro global y se asigna df_filtrado directamente a df para mantener compatibilidad
df_filtrado = df.copy()
if 'tema_dominante' in df_filtrado.columns:
    df_filtrado['tema_dominante'] = df_filtrado['tema_dominante'].replace('transición energética y medio ambiente', 'Transición<br>Energ.')
# ==============================================================================
# Buscador Ciudadano Interactivo de Noticias (EN EL SIDEBAR)
# ==============================================================================
st.sidebar.header("Herramienta de Auditoría Cualitativa")
st.sidebar.markdown("Buscador de Noticias. Módulo de verificación diseñado para asegurar la transparencia y reproducibilidad del estudio. Permite acceder a la base de datos estructurada para consultar el texto íntegro de los artículos periodísticos procesados por los algoritmos de clasificación.")

if not df_filtrado.empty:
    col_fecha = 'fecha_publicacion' if 'fecha_publicacion' in df_filtrado.columns else 'fecha_dia'
    df_filtrado['fecha_str'] = pd.to_datetime(df_filtrado[col_fecha]).dt.strftime('%Y-%m-%d')
    
    medios_disponibles = sorted(df_filtrado['medio_emisor'].dropna().unique())
    medio_buscador = st.sidebar.selectbox("📰 Medio:", medios_disponibles, key="buscador_medio")
    fechas_validas = sorted(df_filtrado[df_filtrado['medio_emisor'] == medio_buscador]['fecha_str'].dropna().unique())
    
    if fechas_validas:
        fecha_buscador = st.sidebar.selectbox("📅 Fecha:", fechas_validas, key="buscador_fecha")
    else:
        fecha_buscador = None
        st.sidebar.warning("No hay fechas válidas.")
        
    df_resultados = df_filtrado[(df_filtrado['medio_emisor'] == medio_buscador) & (df_filtrado['fecha_str'] == fecha_buscador)]
    
    if st.sidebar.button("Buscar Noticia", type="primary", use_container_width=True):
        if df_resultados.empty:
            st.sidebar.error("No se encontraron noticias.")
        else:
            st.sidebar.markdown(f"**Resultados ({len(df_resultados)})**")
            st.sidebar.markdown("---")
            for i, row in df_resultados.reset_index().iterrows():
                titulo = row.get('titular', f'Artículo #{i+1}')
                url = row.get('url', '#')
                cuerpo = row.get('cuerpo', 'Texto no disponible.')
                html_noticia = f"""
                <div style='background-color:#f9f9f9; padding:15px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #2c3e50;'>
                    <h4 style='margin-top:0; color:#1a252f; font-size:14px;'>{titulo}</h4>
                    <p style='color:gray; font-size:10px;'>
                        <b>Medio:</b> {medio_buscador} 
                        {f"| <a href='{url}' target='_blank'>Original</a>" if url and url != '#' else "| Enlace no disponible"}
                    </p>
                    <p style='color:#333; font-size:12px; text-align:justify;'>{cuerpo}</p>
                </div>
                """
                st.sidebar.markdown(html_noticia, unsafe_allow_html=True)
else:
    st.sidebar.info("No hay datos disponibles.")

# ==============================================================================
# Módulo 1: Exploración del Corpus y Cobertura General
# ==============================================================================
st.header("Módulo 1: Exploración del Corpus y Cobertura General")
st.subheader("Panorama General de la Base de Datos")
st.markdown("Un primer vistazo a la base de datos consolidada. La tabla detalla el volumen de noticias, fecha de publicación, medio emisor, titular, autoría y un fragmento extraído.")
st.write(f"**Total de registros listos para EDA:** {len(df_filtrado)}")

if not df_filtrado.empty:
    cols_necesarias = ['fecha_dia', 'titular', 'medio_emisor']
    cols_extra = []
    if 'autor' in df_filtrado.columns: cols_extra.append('autor')
    if 'cuerpo' in df_filtrado.columns: cols_extra.append('cuerpo')
    
    try:
        df_muestra = df_filtrado.groupby('medio_emisor').sample(n=3, replace=False)
    except ValueError:
        df_muestra = df_filtrado.groupby('medio_emisor').apply(lambda x: x.sample(n=min(len(x), 3))).reset_index(drop=True)
        
    df_mostrar = df_muestra[cols_necesarias + cols_extra].copy()
    
    if 'cuerpo' in df_mostrar.columns:
        df_mostrar['cuerpo'] = df_mostrar['cuerpo'].astype(str).str.slice(0, 150) + '...'
        
    df_mostrar = df_mostrar.rename(columns={
        'fecha_dia': 'Fecha', 
        'titular': 'Encabezado', 
        'medio_emisor': 'Medio', 
        'autor': 'Autor', 
        'cuerpo': 'Fragmento de la noticia'
    })
    
    df_mostrar = df_mostrar.sample(frac=1.0).reset_index(drop=True)
    df_mostrar.index.name = '#'
    df_mostrar.index = df_mostrar.index + 1
    
    st.dataframe(df_mostrar, use_container_width=True)

st.subheader("Distribución de Noticias y Red de Autorías")
st.markdown("Distribución volumétrica del corpus, evidenciando un promedio de 250 a 300 publicaciones por medio. A la derecha, una visualización interactiva identifica a los periodistas y autores principales responsables de la cobertura por cada medio.")

col1, col2 = st.columns(2)

with col1:
    # Distribución por Medio
    conteo_medios = df_filtrado['medio'].value_counts().reset_index()
    conteo_medios.columns = ['medio', 'cantidad']
    fig_medios = px.bar(
        conteo_medios, 
        x='cantidad', 
        y='medio', 
        orientation='h',
        title="Distribución de Noticias por Medio",
        color='medio',
        color_discrete_sequence=px.colors.qualitative.Prism,
        text='medio'
    )
    fig_medios.update_traces(texttemplate='<b>%{text}</b>', textposition='auto', insidetextanchor='middle', textfont=dict(size=14))
    fig_medios.update_layout(showlegend=False, xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black'), yaxis={'categoryorder':'total ascending', 'showticklabels': False, 'title': '', 'showline': True, 'linewidth': 1, 'linecolor': 'black'})
    st.plotly_chart(fig_medios, use_container_width=True, theme=None, config={'displayModeBar': False})

with col2:
    # Top 10 Autores (Interactivo por Medio)
    if 'autor' in df_filtrado.columns:
        df_autores = df_filtrado.dropna(subset=['autor']).copy()
        df_autores = df_autores[df_autores['autor'].astype(str).str.strip() != '']
        
        opciones_autores = sorted(df_autores['medio_emisor'].unique().tolist())
        idx_caracol = next((i for i, m in enumerate(opciones_autores) if "CARACOL" in str(m).upper()), 0)
        medio_autor = st.selectbox("Filtrar Autores por Medio:", opciones_autores, index=idx_caracol, key="autor_medio")
        
        df_autores = df_autores[df_autores['medio_emisor'] == medio_autor]
            
        if not df_autores.empty:
            df_autores['autor'] = df_autores['autor'].astype(str).str.split('-').str[0].str.split('|').str[0].str.strip()
            top_autores = df_autores['autor'].value_counts().head(10).reset_index()
            top_autores.columns = ['autor', 'cantidad']
            
            fig_autores = px.bar(
                top_autores, 
                x='cantidad', 
                y='autor', 
                orientation='h',
                title=f"Top 10 Autores ({medio_autor})",
                color='autor',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                text='autor'
            )
            fig_autores.update_traces(texttemplate='<b>%{text}</b>', textposition='auto', insidetextanchor='middle', textfont=dict(size=14))
            fig_autores.update_layout(showlegend=False, xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black'), yaxis={'categoryorder':'total ascending', 'showticklabels': False, 'title': '', 'showline': True, 'linewidth': 1, 'linecolor': 'black'})
            st.plotly_chart(fig_autores, use_container_width=True, theme=None, config={'displayModeBar': False})
        else:
            st.info("No hay autores registrados para este medio.")
    else:
        st.info("Columna 'autor' no disponible en el dataset.")

# ==========================================
# Línea de Tiempo de Cobertura Diaria
# ==========================================
st.subheader("Línea de Tiempo de Cobertura Diaria")
st.markdown("Auditoría visual de la continuidad en la recolección de datos, identificando los días con publicaciones efectivas y posibles vacíos de información.")
# Agrupar por fecha y medio_emisor para obtener el volumen diario
df_timeline = df_filtrado.groupby(['fecha_dia', 'medio_emisor']).size().reset_index(name='cantidad')

fig_timeline = px.scatter(
    df_timeline,
    x='fecha_dia',
    y='medio_emisor',
    size='cantidad',
    color='medio_emisor',
    title="<b>Línea de Tiempo: Cobertura Diaria Estratificada</b>",
    labels={'fecha_dia': 'Fecha de Publicación', 'medio_emisor': 'Medio Emisor', 'cantidad': 'Volumen'},
    color_discrete_sequence=px.colors.qualitative.Set2,
    size_max=15, # [!] Tamaño reducido de 40 a 15
    opacity=0.7  # [!] Opacidad añadida para distinguir puntos superpuestos
)
fig_timeline.for_each_trace(lambda t: t.update(name=f"<b>{t.name}</b>"))
fig_timeline.update_traces(marker=dict(line=dict(width=1, color='white')))
fig_timeline.update_layout(
    xaxis_title="", 
    yaxis_title="",
    plot_bgcolor='white',
    xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black'),
    yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black', showticklabels=False),
    legend_title_text="",
    height=350,
    margin=dict(t=40, b=0, l=0, r=0)
)
fig_timeline.update_xaxes(tickangle=0, tickformat="%b")
st.plotly_chart(fig_timeline, use_container_width=True, theme=None, config={'displayModeBar': False})

# ==========================================
# Distribución Temática Proporcional
# ==========================================
st.subheader("Distribución Temática Proporcional")
st.markdown("Desglose porcentual que revela la estructuración de la agenda y los temas prioritarios seleccionados por cada medio de comunicación.")
df_temas = pd.crosstab(df_filtrado['medio_emisor'], df_filtrado['tema_dominante'], normalize='index') * 100
df_temas_melt = df_temas.reset_index().melt(id_vars='medio_emisor', var_name='tema_dominante', value_name='porcentaje')

fig_temas = px.bar(
    df_temas_melt, 
    x='medio_emisor', 
    y='porcentaje', 
    color='tema_dominante', 
    title='<b>Distribución Temática Proporcional (%)</b>', 
    barmode='group',
    height=450
)
fig_temas.for_each_trace(lambda t: t.update(name=f"<b>{t.name}</b>"))
fig_temas.update_layout(
    bargap=0.15,
    bargroupgap=0.05,
    xaxis_title="",
    yaxis_title="",
    legend_title_text="",
    legend=dict(orientation="h", xanchor="center", x=0.5),
    margin=dict(l=0, r=0, t=60, b=40)
)
fig_temas.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black')
fig_temas.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black')
st.plotly_chart(fig_temas, use_container_width=True, theme=None, config={'displayModeBar': False})

# ==========================================
# Módulo 2: Encuadres de Sentimiento y Agenda Temática
# ==========================================
st.header("Módulo 2: Encuadres de Sentimiento y Agenda Temática")

st.markdown('''
<div style="display: flex; align-items: center; justify-content: flex-start; gap: 20px; margin-bottom: 20px; padding-top: 10px;">
    <div style="display: flex; flex-direction: column; align-items: center; min-width: 80px;">
        <span style="font-size: 0.75em; font-weight: bold; color: #053061;">+1 (Favorable)</span>
        <div style="width: 20px; height: 180px; background: linear-gradient(to bottom, #053061, #4393c3, whitesmoke, #d6604d, #67001f); border: 1px solid #ccc; border-radius: 4px; margin: 4px 0;"></div>
        <span style="font-size: 0.75em; font-weight: bold; color: #67001f;">-1 (Hostil)</span>
    </div>
    <div style="font-size: 0.9em; text-align: justify; flex: 1;">
        <strong>Escala de Medición XLM-RoBERTa</strong><br><br>
        La escala de color muestra el tono emocional de las noticias, calculado de forma automática por un modelo de Inteligencia Artificial. El espectro va desde el <strong>+1.0 (Azul)</strong>, que indica noticias con un lenguaje muy positivo, favorable o de apoyo, hasta el <strong>-1.0 (Rojo)</strong>, que señala un lenguaje de alta hostilidad, crítica directa o alerta de crisis. El valor <strong>0 (Blanco)</strong> representa reportajes estrictamente informativos y neutrales, sin emociones evidentes.
    </div>
</div>
''', unsafe_allow_html=True)

st.subheader("Índice de Sentimiento Editorial")
st.markdown("La siguiente tabla detalla cómo se clasifican los niveles de positividad y negatividad en las noticias. Para cada rango numérico, se presenta su significado periodístico (encuadre editorial) y se incluye un ejemplo real extraído de la base de datos para ilustrar el tono de la cobertura.")

with st.container():
    leyenda_data = {
        "Rango Numérico": [
            "+0.80 a +1.00", "+0.50 a +0.79", "+0.15 a +0.49", 
            "-0.14 a +0.14", 
            "-0.15 a -0.49", "-0.50 a -0.79", "-0.80 a -1.00"
        ],
        "Etiqueta Cualitativa": [
            "Apoyo Explícito / Encomio", "Cobertura Favorable", "Aprobación Sutil / Matiz Positivo",
            "Neutro / Estrictamente Informativo",
            "Tensión Leve / Escepticismo", "Crítica Severa / Desaprobación", "Hostilidad Extrema / Ataque Directo"
        ],
        "Interpretación del Encuadre Editorial": [
            "Cobertura abiertamente laudatoria. Uso intensivo de adjetivos positivos y encuadres de éxito rotundo o triunfo histórico.",
            "Enfoque optimista o constructivo. Se resaltan los logros, avances o virtudes de la gestión de un actor sin llegar a la adulación.",
            "Tono amablemente moderado. Recepción favorable leve, aceptación institucional o registro de expectativas positivas moderadas.",
            "Lenguaje puramente fáctico y descriptivo. Ausencia deliberada de adjetivación calificativa, sesgos emocionales o juicios de valor.",
            "Introducción de dudas, reservas o ironía sutil. El texto registra incomodidad editorial, distanciamiento crítico o cuestionamientos indirectos (aquí se ubican los valores de -0.20 y -0.41).",
            "Cuestionamientos directos a la gestión, moralidad o decisiones del actor político. Uso de lenguaje formal pero con un encuadre marcado de alarma o denuncia.",
            "Encuadre de destrucción reputacional o criminalización (Culpabilidad Activa). Uso intensivo de términos peyorativos, condena absoluta y atribución directa de crisis."
        ]
    }

    medios_usados = set()

    def obtener_ejemplo_real(min_val, max_val, fallback_texto, preferred_medio=None):
        if 'indice_sentimiento' in df_filtrado.columns and 'cuerpo' in df_filtrado.columns:
            subset = df_filtrado[(df_filtrado['indice_sentimiento'] >= min_val) & (df_filtrado['indice_sentimiento'] <= max_val)]
            
            # Intentar primero con el medio preferido
            if preferred_medio:
                subset_pref = subset[subset['medio_emisor'] == preferred_medio]
                if not subset_pref.empty:
                    muestra = subset_pref.sample(1).iloc[0]
                    texto = str(muestra['cuerpo']).strip()
                    medio = str(muestra['medio_emisor'])
                    sent = float(muestra['indice_sentimiento'])
                    if len(texto) > 130:
                        texto = texto[:130] + "..."
                    medios_usados.add(medio)
                    return f"[{medio} | {sent:+.2f}] \"{texto}\""

            # Si no hay medio preferido o el preferido no tiene datos, buscar uno que no se haya usado en la tabla
            subset_no_usados = subset[~subset['medio_emisor'].isin(medios_usados)]
            if not subset_no_usados.empty:
                muestra = subset_no_usados.sample(1).iloc[0]
                texto = str(muestra['cuerpo']).strip()
                medio = str(muestra['medio_emisor'])
                sent = float(muestra['indice_sentimiento'])
                if len(texto) > 130:
                    texto = texto[:130] + "..."
                medios_usados.add(medio)
                return f"[{medio} | {sent:+.2f}] \"{texto}\""
            
            # Si todos fallan (ya se usaron todos los medios), tomar cualquiera
            if not subset.empty:
                muestra = subset.sample(1).iloc[0]
                texto = str(muestra['cuerpo']).strip()
                medio = str(muestra['medio_emisor'])
                sent = float(muestra['indice_sentimiento'])
                if len(texto) > 130:
                    texto = texto[:130] + "..."
                return f"[{medio} | {sent:+.2f}] \"{texto}\""
                
        return fallback_texto

    leyenda_data["Ejemplo Ilustrativo de Cobertura"] = [
        obtener_ejemplo_real(0.80, 1.00, "Ejemplo Genérico: El histórico acuerdo liderado por el mandatario representa un triunfo sin precedentes..."),
        obtener_ejemplo_real(0.50, 0.79, "Ejemplo Genérico: La comitiva logró consolidar importantes alianzas estratégicas internacionales...", preferred_medio="ELESPECTADOR"),
        obtener_ejemplo_real(0.15, 0.49, "Ejemplo Genérico: El nuevo plan de infraestructura vial ha sido recibido con optimismo...", preferred_medio="CARACOL"),
        obtener_ejemplo_real(-0.14, 0.14, "Ejemplo Genérico: El comité de presupuesto se reunirá el martes para evaluar la agenda...", preferred_medio="ELTIEMPO"),
        obtener_ejemplo_real(-0.49, -0.15, "Ejemplo Genérico: Varios sectores manifestaron reservas ante la viabilidad técnica...", preferred_medio="LASILLAVACIA"),
        obtener_ejemplo_real(-0.79, -0.50, "Ejemplo Genérico: La alarmante falta de planeación desató una profunda crisis institucional...", preferred_medio="RCN"),
        obtener_ejemplo_real(-1.00, -0.80, "Ejemplo Genérico: El nefasto manejo de los recursos públicos demuestra una negligencia sistemática...", preferred_medio="SEMANA")
    ]

    st.dataframe(pd.DataFrame(leyenda_data), hide_index=True, use_container_width=True)

st.subheader("Visualización de Sentimientos y Mapa de Calor")
st.markdown("Contraste de los temas abordados frente a su encuadre predominante (positivo, negativo o neutro). Adicionalmente, se presenta el Mapa de Calor Temático. (Nota metodológica: Casos de sarcasmo detectados e imputados en vivo: 0).")
if 'prob_POS' in df_filtrado.columns and 'prob_NEG' in df_filtrado.columns and 'prob_NEU' in df_filtrado.columns:
    df_sentiments = df_filtrado.groupby('tema_dominante')[['prob_POS', 'prob_NEG', 'prob_NEU']].mean().reset_index()
    df_sent_melted = df_sentiments.melt(id_vars='tema_dominante', var_name='Sentimiento', value_name='Probabilidad Media')
    df_sent_melted['Sentimiento'] = df_sent_melted['Sentimiento'].map({'prob_POS': 'Positivo', 'prob_NEG': 'Negativo', 'prob_NEU': 'Neutro'})
    
    fig_sentiments = px.bar(
        df_sent_melted,
        x='tema_dominante',
        y='Probabilidad Media',
        color='Sentimiento',
        barmode='group',
        title="<b>Sentimiento Promedio por Temática Dominante</b>",
        color_discrete_map={'Positivo': '#0571b0', 'Negativo': '#ca0020', 'Neutro': '#cccccc'}
    )
    fig_sentiments.update_traces(texttemplate='<b>%{y:.2f}</b>', textposition='outside')
    fig_sentiments.for_each_trace(lambda t: t.update(
        name=f"<b>{t.name}</b>",
        textfont=dict(color=t.marker.color)
    ))
    fig_sentiments.update_layout(
        xaxis_title="", 
        yaxis_title="",
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black'),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=False, ticks='inside', ticklen=8, tickcolor='black', showticklabels=False),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    st.plotly_chart(fig_sentiments, use_container_width=True, theme=None, config={'displayModeBar': False})
else:
    st.info("No se encontraron columnas de probabilidad de sentimiento en el dataset.")

# ==============================================================================
# Mapa de Calor: Encuadre Temático (Escala Universal -1 a +1)
# ==============================================================================
# El Mapa de Calor ahora está dentro de la subsección anterior

if 'indice_sentimiento' in df_filtrado.columns and 'emocion_dominante' in df_filtrado.columns:
    # 1. Regla de Disonancia Lingüística (Sarcasmo / Alegría Maliciosa)
    import numpy as np
    condicion_disonancia = (df_filtrado['prob_NEG'] >= 0.65) & (df_filtrado['emocion_dominante'] == 'Felicidad/Joy')
    df_filtrado['Firma_Emocional_Ajustada'] = np.where(condicion_disonancia, 'Sarcasmo / Alegría Maliciosa', df_filtrado['emocion_dominante'])
    st.caption("Nota: En la base de datos no se detectaron ni imputaron casos de sarcasmo.")

    # 2. Construcción de la matriz de encuadre
    df_enc = df_filtrado.groupby(['medio_emisor', 'tema_dominante'])['indice_sentimiento'].mean().reset_index()
    matriz_enc = df_enc.pivot(index='medio_emisor', columns='tema_dominante', values='indice_sentimiento').fillna(0)

    # 3. Renderizado del Mapa de Calor
    fig_heatmap = px.imshow(
        matriz_enc, 
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        title='Encuadre Temático: Promedio de Sentimiento por Medio y Tema',
        labels=dict(x="", y="", color=""),
        aspect="auto",
        height=600
    )
    fig_heatmap.update_layout(coloraxis_colorbar=dict(thickness=10, title=""),
        plot_bgcolor='white',
        xaxis=dict(tickangle=45, showgrid=True, gridcolor='whitesmoke'),
        yaxis=dict(showgrid=True, gridcolor='whitesmoke')
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, theme=None, config={'displayModeBar': False})
else:
    st.info("No se encontraron las columnas necesarias (indice_sentimiento, emocion_dominante) para el mapa de calor.")

# ==============================================================================
# Matriz de Carriles Temáticos (Agenda Matrix)
# ==============================================================================
st.subheader("Matriz de Agenda (Top Temas Prioritarios)")
st.markdown("Evolución longitudinal de las temáticas por medio emisor. El color de la esfera indica la polaridad del encuadre; la ausencia de esfera indica nula cobertura, un diámetro menor equivale a una noticia y un diámetro mayor representa múltiples publicaciones.")

import plotly.graph_objects as go

# Validar que tengamos la llave correcta para contar
col_conteo = 'uuid_doc' if 'uuid_doc' in df_filtrado.columns else 'url'

if 'indice_sentimiento' in df_filtrado.columns:
    df_agrupado = df_filtrado.groupby(['fecha_dia', 'medio_emisor', 'tema_dominante']).agg(
        volumen_noticias=(col_conteo, 'count'),
        sentimiento_promedio=('indice_sentimiento', 'mean')
    ).reset_index()

    if 'len_cuerpo_words' in df_filtrado.columns:
        long_media = df_filtrado.groupby(['fecha_dia', 'medio_emisor', 'tema_dominante'])['len_cuerpo_words'].mean().reset_index()
        df_agrupado = pd.merge(df_agrupado, long_media, on=['fecha_dia', 'medio_emisor', 'tema_dominante'], how='left')
    else:
        df_agrupado['len_cuerpo_words'] = 0

    df_agrupado = df_agrupado.dropna(subset=['volumen_noticias', 'sentimiento_promedio'])
    
    medios_unicos = sorted(df_agrupado['medio_emisor'].unique())
    if medios_unicos:
        medio_seleccionado = st.selectbox("Selecciona el Medio para ver su Agenda Temática:", medios_unicos)
        
        df_medio = df_agrupado[df_agrupado['medio_emisor'] == medio_seleccionado].copy()
        
        if not df_medio.empty:
            tema_vol = df_medio.groupby('tema_dominante')['volumen_noticias'].sum().reset_index()
            top_temas = tema_vol.nlargest(8, 'volumen_noticias')['tema_dominante'].tolist()
            
            df_top = df_medio[df_medio['tema_dominante'].isin(top_temas)].copy()
            df_top['tema_dominante'] = pd.Categorical(df_top['tema_dominante'], categories=top_temas[::-1], ordered=True)
            df_top = df_top.sort_values(['tema_dominante', 'fecha_dia'])
            
            max_vol = df_top['volumen_noticias'].max()
            ref = 2. * max_vol / (50.**2) if max_vol > 0 else 1
            
            fig_agenda = go.Figure()
            
            hover_texts = []
            for _, row in df_top.iterrows():
                txt = (
                    f"<b>{row['medio_emisor']}</b><br>"
                    f"📅 Fecha: {str(row['fecha_dia'])[:10]}<br>"
                    f"🎯 Tema: <b>{row['tema_dominante']}</b><br>"
                    f"📰 Volumen: {int(row['volumen_noticias'])} noticias<br>"
                    f"🔴🔵 Sentimiento: {round(row['sentimiento_promedio'], 3)}<br>"
                    f"📝 Longitud: {int(row['len_cuerpo_words'])} palabras"
                )
                hover_texts.append(txt)
            
            fig_agenda.add_trace(go.Scatter(
                x=df_top['fecha_dia'].astype(str).tolist(),
                y=df_top['tema_dominante'].astype(str).tolist(),
                mode='markers',
                text=hover_texts,
                hoverinfo='text',
                marker=dict(
                    size=df_top['volumen_noticias'].astype(float).tolist(), 
                    sizemode='area', sizeref=ref, sizemin=6,
                    color=df_top['sentimiento_promedio'].astype(float).tolist(), 
                    coloraxis="coloraxis",
                    line=dict(width=1, color='rgba(50, 50, 50, 0.5)')
                ),
                name=medio_seleccionado
            ))
            
            fig_agenda.update_layout(coloraxis_colorbar=dict(thickness=10, title=""),
                title=f"Matriz de Agenda para {medio_seleccionado} (Tamaño=Volumen | Color=Sentimiento)",
                xaxis_title="",
                yaxis_title="",
                height=600,
                plot_bgcolor='white', 
                xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5), 
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1, colorbar=dict(title="Sentimiento")) 
            )
            st.plotly_chart(fig_agenda, use_container_width=True, theme=None, config={'displayModeBar': False})
else:
    st.info("Faltan columnas de sentimiento para la matriz de agenda.")

# ==============================================================================
# Módulo 3: Dinámicas Candidato-Medio (Panorama General)
# ==============================================================================
st.header("Módulo 3: Dinámicas Candidato-Medio")

cols_existentes = [col for col in df_filtrado.columns if '_sesgo_detectado' in col]

if cols_existentes:
    cols_to_melt = ['fecha_dia', 'medio_emisor']
    if 'len_cuerpo_words' in df_filtrado.columns:
        cols_to_melt.append('len_cuerpo_words')
        
    df_melt_cand = df_filtrado.melt(
        id_vars=cols_to_melt, 
        value_vars=cols_existentes, 
        var_name='candidato', 
        value_name='sesgo'
    )
    
    df_melt_cand = df_melt_cand.dropna(subset=['sesgo'])
    df_melt_cand['sesgo'] = df_melt_cand['sesgo'].astype(str).str.strip()
    valores_ignorados = ['None', 'nan', 'Sin Mencion', 'Neutral', '']
    df_melt_cand = df_melt_cand[~df_melt_cand['sesgo'].isin(valores_ignorados)]
    df_melt_cand['candidato'] = df_melt_cand['candidato'].str.replace('_sesgo_detectado', '').str.replace('_', ' ')
    
    st.write(f"📊 **Total de menciones válidas a graficar:** {len(df_melt_cand)}")
    
    if not df_melt_cand.empty:
        # 1. TREEMAP: Volumen total
        df_vol_cand = df_melt_cand.groupby('candidato').size().reset_index(name='menciones')
        fig_treemap = px.treemap(df_vol_cand, path=['candidato'], values='menciones', title='Panorama de Menciones', height=600)
        fig_treemap.update_traces(textinfo="label+value")
        st.plotly_chart(fig_treemap, use_container_width=True, theme=None, config={'displayModeBar': False})
            
        # 2. DISPERSIÓN: Frecuencia vs Extensión
        st.subheader("Frecuencia vs. Extensión")
        st.markdown("Análisis de menciones por actor político. Correlación entre la frecuencia de publicación sobre un candidato frente a la extensión promedio (cantidad de palabras) de los artículos que lo referencian.")
        if 'len_cuerpo_words' in df_filtrado.columns:
            df_disp = df_melt_cand.groupby('candidato').agg(frecuencia=('sesgo', 'count'), longitud=('len_cuerpo_words', 'mean')).reset_index()
            fig_scatter_cand = px.scatter(df_disp, x='frecuencia', y='longitud', text='candidato', size='frecuencia', title='Frecuencia vs. Extensión', height=600)
            fig_scatter_cand.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter_cand, use_container_width=True, theme=None, config={'displayModeBar': False})
        else:
            st.info("No se puede graficar dispersión por falta de la columna de longitud del cuerpo.")

        # 3. BARRAS AGRUPADAS: Perfil Editorial del Medio
        st.subheader("Perfil Editorial (Volumen de Cobertura)")
        st.markdown("Cuantificación de la atención prestada por cada medio a los candidatos. El conteo considera la co-ocurrencia de entidades, registrando los casos donde un artículo menciona a múltiples actores simultáneamente.")
        df_medios_cand = df_melt_cand.groupby(['medio_emisor', 'candidato']).size().reset_index(name='menciones')
        
        fig_barras = px.bar(
            df_medios_cand, 
            x='medio_emisor',      
            y='menciones', 
            color='candidato',     
            title='Perfil Editorial: Volumen de Cobertura',
            text_auto=True,
            barmode='group',       
            height=500
        )
        
        fig_barras.update_layout(
            xaxis={'categoryorder':'total descending'}, 
            plot_bgcolor='white',
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_barras, use_container_width=True, theme=None, config={'displayModeBar': False})
        
    else:
        st.warning("No hay datos suficientes después de la limpieza para graficar el módulo de candidatos.")
else:
    st.error("No se encontraron columnas de sesgo de candidatos en el dataset.")

# ==============================================================================
# Mapa de Calor Cruzado: Medios vs Candidatos vs Encuadres
# ==============================================================================
st.subheader("Mapa de Calor Evolutivo")
st.markdown("Evolución del índice de polaridad (escala de +1 a -1) durante los seis meses de estudio. Revela el tema dominante asociado al candidato y el volumen de noticias, filtrando exclusivamente las relaciones con más de 4 publicaciones mensuales.")

df_tiempo = df_filtrado.copy()
if not df_tiempo.empty:
    df_tiempo['fecha'] = pd.to_datetime(df_tiempo['fecha_publicacion']).dt.tz_localize(None)
    df_tiempo['Mes'] = df_tiempo['fecha'].dt.strftime('%Y-%m')

    candidatos_base = [
        'Gustavo_Petro', 'Iván_Cepeda', 'Abelardo_de_la_Espriella',
        'Paloma_Valencia', 'Sergio_Fajardo', 'Claudia_López', 'Álvaro_Uribe_Vélez'
    ]

    frames = []
    for cand in candidatos_base:
        col_sent = f"{cand}_indice_sentimiento"
        col_sesgo = f"{cand}_sesgo_detectado"
        
        if col_sent in df_tiempo.columns and col_sesgo in df_tiempo.columns:
            temp = df_tiempo[['medio_emisor', 'Mes', 'tema_dominante', col_sent, col_sesgo]].copy()
            temp.rename(columns={col_sent: 'Sentimiento', col_sesgo: 'Encuadre'}, inplace=True)
            temp['Candidato'] = cand.replace('_', ' ')
            frames.append(temp)

    if frames:
        df_melted_hm = pd.concat(frames, ignore_index=True)
        df_melted_hm['Sentimiento'] = df_melted_hm['Sentimiento'].replace({0.0: np.nan})
        df_melted_hm = df_melted_hm.dropna(subset=['Sentimiento'])

        import textwrap
        def calcular_mutacion(x):
            moda_tema = x['tema_dominante'].mode()
            tema_principal = moda_tema.iloc[0] if not moda_tema.empty else 'Sin Datos'
            
            if tema_principal == 'Sin Datos':
                return pd.Series({'Sentimiento_Promedio': np.nan, 'Texto_Anotacion': ''})
            
            df_tema_ganador = x[x['tema_dominante'] == tema_principal]
            volumen_real = len(df_tema_ganador)
            
            if volumen_real <= 4:
                return pd.Series({'Sentimiento_Promedio': np.nan, 'Texto_Anotacion': ''})
            
            sesgos_validos = df_tema_ganador['Encuadre'].astype(str).str.strip()
            sesgos_validos = sesgos_validos.replace({'None': np.nan, 'nan': np.nan, 'Sin Mencion': np.nan, 'Neutral': np.nan, '': np.nan}).dropna()
            
            moda_encuadre = sesgos_validos.mode()
            encuadre_principal = moda_encuadre.iloc[0] if not moda_encuadre.empty else 'Sin Rol Fijo'
            
            sentimiento_prom = df_tema_ganador['Sentimiento'].mean()
            tema_corto = textwrap.shorten(tema_principal, width=20, placeholder="...")
            texto_celda = f"<b>T:</b> {tema_corto}<br><b>E:</b> {encuadre_principal}<br>(n={volumen_real})<br>[{sentimiento_prom:+.2f}]"
            
            return pd.Series({'Sentimiento_Promedio': sentimiento_prom, 'Texto_Anotacion': texto_celda})

        df_agrup_hm = df_melted_hm.groupby(['medio_emisor', 'Candidato', 'Mes']).apply(calcular_mutacion).reset_index()

        meses_unicos_hm = sorted(df_agrup_hm['Mes'].unique())
        candidatos_unicos_hm = sorted(df_agrup_hm['Candidato'].unique())
        medios_unicos_hm = sorted(df_agrup_hm['medio_emisor'].unique())

        if medios_unicos_hm:
            medio_hm = st.selectbox("Selecciona el Medio para ver Evolución Mensual:", medios_unicos_hm, key="hm_medio")
            
            df_m = df_agrup_hm[df_agrup_hm['medio_emisor'] == medio_hm]
            
            if not df_m.empty:
                matriz_z = df_m.pivot(index='Candidato', columns='Mes', values='Sentimiento_Promedio').reindex(index=candidatos_unicos_hm, columns=meses_unicos_hm)
                matriz_text = df_m.pivot(index='Candidato', columns='Mes', values='Texto_Anotacion').reindex(index=candidatos_unicos_hm, columns=meses_unicos_hm).fillna('')
                
                fig_hm = go.Figure(data=go.Heatmap(
                    z=matriz_z.values,
                    x=meses_unicos_hm,
                    y=candidatos_unicos_hm,
                    text=matriz_text.values,
                    texttemplate="%{text}",      
                    hoverinfo="text",            
                    coloraxis="coloraxis",       
                    xgap=2, ygap=2               
                ))
                
                fig_hm.update_layout(coloraxis_colorbar=dict(thickness=10, title=""),
                    title=f"Evolución del Encuadre Mensual: <b>{medio_hm}</b><br><sup>(Filtro riguroso: >4 noticias/mes | Tema, Rol, Volumen y Tono)</sup>",
                    xaxis_title="",
                    yaxis_title="",
                    height=1000,        
                    coloraxis=dict(
                        colorscale='RdBu', 
                        cmin=-1, cmax=1, 
                        colorbar=dict(title="Tono Editorial<br>(-1 Ataque | +1 Defensa)")
                    ),
                    plot_bgcolor='white',
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig_hm, use_container_width=True, theme=None, config={'displayModeBar': False})
            else:
                st.info("No hay datos suficientes (que pasen el filtro de volumen) para este medio.")
    else:
         st.warning("No se encontraron columnas de sentimiento de candidatos.")

# ==============================================================================
# Módulo 4: Asimetría de Encuadres y Hostilidad
# ==============================================================================
st.header("Módulo 4: Asimetría de Encuadres y Hostilidad")

st.subheader("Resumen de Encuadres Mapeados")
st.markdown("""
**Guía Metodológica de Hostilidad:**

**Culpabilidad Activa**: Sentimiento Negativo + Rol de Sujeto Activo. El artículo posee carga tóxica y el actor político es el ejecutor de la acción.

**Victimización**: Sentimiento Negativo + Rol de Sujeto Pasivo. Carga tóxica donde el actor político es el receptor de la acción.

**Contexto Adverso**: Sentimiento altamente negativo (Probabilidad > 70%) + Rol Sintáctico Desconocido. Mención del político dentro de un entorno crítico o de extrema negatividad, generando una asociación perjudicial implícita.
""")

# ==============================================================================
# Asimetría de Encuadres (Culpabilidad, Victimización y Ataque)
# ==============================================================================
st.subheader("Volumen Total de Hostilidad")
st.markdown("Consolidación de la asimetría editorial. Sumatoria de encuadres de Culpabilidad Activa, Victimización y Contexto Adverso dirigidos por los medios hacia cada candidato.")

cols_existentes_asi = [col for col in df_filtrado.columns if '_sesgo_detectado' in col]

if cols_existentes_asi:
    col_conteo_asi = 'uuid_doc' if 'uuid_doc' in df_filtrado.columns else 'url'
    df_melt_asi = df_filtrado.melt(id_vars=[col_conteo_asi, 'medio_emisor'], value_vars=cols_existentes_asi, var_name='candidato', value_name='sesgo')
    
    df_melt_asi['candidato'] = df_melt_asi['candidato'].str.replace('_sesgo_detectado', '').str.replace('_', ' ')
    df_melt_asi['sesgo'] = df_melt_asi['sesgo'].astype(str).str.strip().str.title()
    
    encuadres_hostiles = [
        'Culpabilidad Activa', 
        'Victimización', 
        'Victimizacion',
        'Contexto Adverso'
    ]
    
    df_hostil = df_melt_asi[df_melt_asi['sesgo'].isin(encuadres_hostiles)]
    asimetria = df_hostil.groupby(['medio_emisor', 'candidato']).size().reset_index(name='frecuencia')
    
    if not asimetria.empty:
        matriz_asi = asimetria.pivot(index='medio_emisor', columns='candidato', values='frecuencia').fillna(0)
        total_hostil = asimetria['frecuencia'].sum()
        
        fig_asi = px.imshow(
            matriz_asi, 
            text_auto=True, 
            color_continuous_scale='OrRd', 
            title=f'Asimetría de Encuadres: Volumen Total de Hostilidad (N={total_hostil})',
            labels=dict(x="", y="", color=""),
            aspect="auto",
            height=900
        )
        
        fig_asi.update_layout(coloraxis_colorbar=dict(thickness=10, title=""),
            plot_bgcolor='white',
            xaxis=dict(tickangle=45, showgrid=True, gridcolor='whitesmoke'),
            yaxis=dict(showgrid=True, gridcolor='whitesmoke')
        )
        
        st.plotly_chart(fig_asi, use_container_width=True, theme=None, config={'displayModeBar': False})
        
        with st.expander("📊 Resumen de Encuadres Mapeados (Datos Exactos)"):
            st.dataframe(df_hostil['sesgo'].value_counts().reset_index(name='Cantidad').rename(columns={'sesgo': 'Encuadre Hostil'}))
    else:
        st.info("No se encontraron encuadres hostiles para graficar después de aplicar los filtros.")
else:
    st.error("No se encontraron las columnas de '_sesgo_detectado' para la asimetría.")


# ==============================================================================
# Macro Tendencias de Encuadres Hostiles (SMA 7)
# ==============================================================================
st.subheader("Datos Exactos y Promedios")
st.markdown("Promedio temporal de encuadres de ataque emitidos por los medios hacia los distintos actores políticos.")

if 'df_melt_cand' in locals() and not df_melt_cand.empty:
    ataques_keywords = ['culpabilidad activa', 'victimización', 'victimizacion']
    df_ataques_sma = df_melt_cand[df_melt_cand['sesgo'].str.lower().isin(ataques_keywords)]
    
    if not df_ataques_sma.empty:
        df_ts_cand = df_ataques_sma.groupby(['fecha_dia', 'candidato']).size().unstack().fillna(0)
        df_sma = df_ts_cand.rolling(window=7, min_periods=1).mean().reset_index().melt(id_vars='fecha_dia', value_name='sma_7')
        
        fig_sma = px.line(
            df_sma, 
            x='fecha_dia', 
            y='sma_7', 
            color='candidato', 
            title='Macro Tendencias: Promedio Móvil (7 días) de Encuadres de Ataque', 
            height=600
        )
        fig_sma.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='whitesmoke'),
            yaxis=dict(showgrid=True, gridcolor='whitesmoke'),
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_sma, use_container_width=True, theme=None, config={'displayModeBar': False})
    else:
        st.info("No se detectaron ataques o victimizaciones para graficar tendencias.")
else:
    st.error("No se encontró la base de datos pre-procesada de candidatos (Módulo 3) requerida para las tendencias.")

st.subheader("Línea de Tiempo de Alianzas Editoriales")
st.markdown("Algoritmo de detección de coaliciones. Identifica periodos donde dos o más medios convergen temáticamente con un índice de sentimiento inferior a -0.38, indicador de tensión leve, distanciamiento crítico o cuestionamientos indirectos.")

if 'indice_sentimiento' in df_filtrado.columns:
    df_ataques = df_filtrado[df_filtrado['indice_sentimiento'] <= -0.38].copy()
    
    if not df_ataques.empty:
        df_ataques['fecha_dt'] = pd.to_datetime(df_ataques['fecha_dia'])
        df_ataques['Mes'] = df_ataques['fecha_dt'].dt.strftime('%Y-%m')
        
        top_temas_ataques = df_ataques.groupby('tema_dominante').size().nlargest(10).index.tolist()
        if 'internacional' not in top_temas_ataques:
            top_temas_ataques.insert(0, 'internacional')
        
        default_index = top_temas_ataques.index('internacional')
        tema_alianza = st.selectbox("Selecciona la Perspectiva Temática para evaluar Coaliciones:", top_temas_ataques, index=default_index)
        
        df_filtro_alianza = df_ataques[df_ataques['tema_dominante'] == tema_alianza]
            
        alianzas_maximas = []
        for (mes, dia), grupo in df_filtro_alianza.groupby(['Mes', 'fecha_dt']):
            medios_dia = sorted(grupo['medio_emisor'].unique())
            n_medios = len(medios_dia)
            if n_medios >= 2:
                avg_sent = grupo['indice_sentimiento'].mean()
                nombre_alianza = " + ".join(medios_dia)
                alianzas_maximas.append({
                    'Mes': mes, 
                    'Coalicion': nombre_alianza, 
                    'num_medios': n_medios,
                    'sent_neg': avg_sent,
                    'fecha_evento': dia
                })
                
        if alianzas_maximas:
            df_alianzas = pd.DataFrame(alianzas_maximas)
            
            def format_fechas_exactas(fechas_series):
                fechas_ordenadas = sorted(fechas_series)
                fechas_str = [d.strftime('%Y-%m-%d') for d in fechas_ordenadas]
                return "<br>          &#8226; " + "<br>          &#8226; ".join(fechas_str)

            df_agrup_alianzas = df_alianzas.groupby(['Coalicion', 'num_medios', 'Mes']).agg(
                dias_sync=('sent_neg', 'count'),       
                avg_sentimiento=('sent_neg', 'mean'),  
                fecha_str=('fecha_evento', format_fechas_exactas) 
            ).reset_index()
            
            frecuencia_historica = df_agrup_alianzas.groupby('Coalicion')['dias_sync'].sum().reset_index(name='total_historico')
            df_agrup_alianzas = df_agrup_alianzas.merge(frecuencia_historica, on='Coalicion')
            
            es_frecuente = df_agrup_alianzas['total_historico'] >= 2
            es_multialianza = df_agrup_alianzas['num_medios'] >= 3
            df_agrup_alianzas = df_agrup_alianzas[es_frecuente | es_multialianza]
            
            if not df_agrup_alianzas.empty:
                orden_y = df_agrup_alianzas[['Coalicion', 'num_medios']].drop_duplicates().sort_values(
                    by=['num_medios', 'Coalicion'], ascending=[True, False]
                )['Coalicion'].tolist()
                
                tamaños_burbujas = [min(max(d * 10, 15), 45) for d in df_agrup_alianzas['dias_sync']]
                
                hover_texts = []
                for _, row in df_agrup_alianzas.iterrows():
                    txt = (
                        f"<b>Coalición:</b> {row['Coalicion']}<br>"
                        f"<b>Mes:</b> {row['Mes']}<br>"
                        f"<b>Días de Ataque:</b> {row['dias_sync']}<br>"
                        f"<b>Sentimiento Promedio:</b> {row['avg_sentimiento']:.2f}<br>"
                        f"<b>Fechas Exactas:</b>{row['fecha_str']}"
                    )
                    hover_texts.append(txt)

                fig_alianzas = go.Figure()
                fig_alianzas.add_trace(go.Scatter(
                    x=df_agrup_alianzas['Mes'],
                    y=df_agrup_alianzas['Coalicion'],
                    mode='markers+text',
                    text=df_agrup_alianzas['dias_sync'], 
                    textfont=dict(color='white', size=11, weight='bold', family="Arial Black"),
                    marker=dict(
                        size=tamaños_burbujas,
                        color=df_agrup_alianzas['avg_sentimiento'], 
                        coloraxis="coloraxis", 
                        opacity=1.0, 
                        line=dict(width=1.5, color='rgba(0, 0, 0, 0.1)')
                    ),
                    textposition='middle center',
                    hovertext=hover_texts,
                    hoverinfo='text'
                ))
                
                import textwrap
                opcion_corta = textwrap.shorten(tema_alianza, width=45, placeholder="...")
                
                fig_alianzas.update_layout(
                    title=f"Evolución de Mega-Alianzas<br><sup>Tema: <b>{opcion_corta}</b> (Tamaño = Días Coordinados | Color = Sentimiento -1 a +1)</sup>",
                    xaxis_title="",
                    yaxis_title="",
                    yaxis=dict(categoryorder='array', categoryarray=orden_y, showgrid=True, gridcolor='white'),
                    height=800,
                    margin=dict(l=400, r=50, t=100, b=50), 
                    plot_bgcolor='#fafafa', 
                    xaxis=dict(showgrid=True, gridcolor='white', tickangle=45),
                    coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1, colorbar=dict(title="Sentimiento<br>Editorial"))
                )
                st.plotly_chart(fig_alianzas, use_container_width=True, theme=None, config={'displayModeBar': False})
            else:
                st.info("No se encontraron coaliciones duras (≥ 2 medios frecuentes o ≥ 3 medios sincronizados) para este tema.")
        else:
            st.info("No hay eventos coordinados para el tema seleccionado.")
    else:
        st.info("No hay registros clasificados como ataque (Sentimiento <= -0.38) con los filtros actuales.")
else:
    st.info("No se encontró la columna de sentimiento para ejecutar el análisis de hostilidad.")

# ==============================================================================
# Módulo 5: Análisis Causal (Efecto Dominó)
# ==============================================================================
st.header("Módulo 5: Análisis Causal (Efecto Dominó y Gatillo)")
st.subheader("Efecto Dominó")
st.markdown("Análisis temporal mediante doble eje anclado. Contrasta la suma de la negatividad dirigida registrada en el periodo previo (t-1) frente al volumen exacto de ataques directos ejecutados (barras) en el mes/semana de estudio.")

cols_existentes_causal = [col for col in df_filtrado.columns if '_sesgo_detectado' in col]

if cols_existentes_causal:
    df_melt_causal = df_filtrado.melt(id_vars=['fecha_publicacion', 'prob_NEG'], 
                      value_vars=cols_existentes_causal, var_name='candidato', value_name='sesgo')
    
    df_melt_causal['candidato'] = df_melt_causal['candidato'].str.replace('_sesgo_detectado', '').str.replace('_', ' ')
    df_melt_causal['sesgo_limpio'] = df_melt_causal['sesgo'].astype(str).str.strip().str.title()
    
    valores_ignorados = ['None', 'Nan', 'Sin Mencion', 'Neutral', '']
    df_melt_causal = df_melt_causal[~df_melt_causal['sesgo_limpio'].isin(valores_ignorados)]
    
    encuadres_hostiles_causal = ['Culpabilidad Activa', 'Victimización', 'Victimizacion', 'Contexto Adverso']
    import numpy as np
    df_melt_causal['Toxicidad_Dirigida'] = np.where(
        df_melt_causal['sesgo_limpio'].isin(encuadres_hostiles_causal),
        df_melt_causal['prob_NEG'], 
        0
    )
    
    df_melt_causal['Es_Ataque'] = (df_melt_causal['sesgo_limpio'] == 'Culpabilidad Activa').astype(int)
    df_melt_causal['fecha_semana'] = df_melt_causal['fecha_publicacion'].dt.to_period('W').dt.start_time
    
    df_agrupado_causal = df_melt_causal.groupby(['candidato', 'fecha_semana']).agg(
        Negatividad_Semana=('Toxicidad_Dirigida', 'sum'), 
        Total_Ataques_Semana=('Es_Ataque', 'sum') 
    ).reset_index()
    
    semanas_unicas = sorted(df_melt_causal['fecha_semana'].dropna().unique())
    candidatos_unicos_causal = df_melt_causal['candidato'].unique()
    
    grid = pd.MultiIndex.from_product([candidatos_unicos_causal, semanas_unicas], names=['candidato', 'fecha_semana']).to_frame(index=False)
    df_cand_sem = pd.merge(grid, df_agrupado_causal, on=['candidato', 'fecha_semana'], how='left')
    
    df_cand_sem['Negatividad_Semana'] = df_cand_sem['Negatividad_Semana'].fillna(0)
    df_cand_sem['Total_Ataques_Semana'] = df_cand_sem['Total_Ataques_Semana'].fillna(0)
    
    df_cand_sem = df_cand_sem.sort_values(['candidato', 'fecha_semana'])
    df_cand_sem['Negatividad_Semana_Pasada'] = df_cand_sem.groupby('candidato')['Negatividad_Semana'].shift(1)
    
    df_cand_sem = df_cand_sem.dropna(subset=['Negatividad_Semana_Pasada'])
    
    candidatos_atacados = df_cand_sem.groupby('candidato')['Total_Ataques_Semana'].sum()
    candidatos_validos = candidatos_atacados[candidatos_atacados > 0].index.tolist()

    if candidatos_validos:
        max_toxicidad_global = df_cand_sem['Negatividad_Semana_Pasada'].max()
        max_ataques_global = df_cand_sem['Total_Ataques_Semana'].max()
        
        limite_y_izquierdo = max_toxicidad_global * 1.1 if max_toxicidad_global > 0 else 1
        limite_y_derecho = max_ataques_global * 1.1 if max_ataques_global > 0 else 1

        default_idx = candidatos_validos.index("Sergio Fajardo") if "Sergio Fajardo" in candidatos_validos else 0
        cand_efecto = st.selectbox("Selecciona el Actor Político Objetivo:", candidatos_validos, index=default_idx)
        
        df_sub = df_cand_sem[df_cand_sem['candidato'] == cand_efecto]
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import textwrap
        
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_time.add_trace(
            go.Scatter(
                x=df_sub['fecha_semana'], 
                y=df_sub['Negatividad_Semana_Pasada'],
                mode='lines+markers',
                name=f'Ola Hostil a {textwrap.shorten(cand_efecto, 10)} (t-1)',
                line=dict(color='gray', width=3, dash='solid'),
                marker=dict(color='black', size=6),
                fill='tozeroy', fillcolor='rgba(200, 200, 200, 0.3)',
                hovertemplate="Semana: %{x}<br>Toxicidad Acumulada: %{y:.1f}<extra></extra>"
            ),
            secondary_y=False,
        )

        fig_time.add_trace(
            go.Bar(
                x=df_sub['fecha_semana'], 
                y=df_sub['Total_Ataques_Semana'],
                name=f'Ataques Reales a {textwrap.shorten(cand_efecto, 10)} (t)',
                marker_color='firebrick',
                opacity=0.8,
                width=1000*3600*24*3,
                hovertemplate="Semana: %{x}<br>Total de Ataques: %{y}<extra></extra>"
            ),
            secondary_y=True,
        )

        fig_time.update_layout(
            title=f"El Efecto Dominó (Línea de Tiempo Causal)<br><sup>Objetivo Específico: <b>{cand_efecto}</b></sup>",
            xaxis_title="",
            height=550,
            plot_bgcolor='white',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig_time.update_yaxes(
            title_text="Volumen de Toxicidad (Suma de Negatividad)", 
            secondary_y=False, showgrid=True, gridcolor='whitesmoke', 
            range=[0, limite_y_izquierdo] 
        )
        fig_time.update_yaxes(
            title_text="Cantidad Exacta de Ataques", 
            secondary_y=True, showgrid=False, 
            range=[0, limite_y_derecho] 
        )

        st.plotly_chart(fig_time, use_container_width=True, theme=None, config={'displayModeBar': False})
    else:
        st.info("No hay candidatos con suficientes datos o ataques para graficar el efecto dominó.")
else:
    st.error("No se encontraron las columnas de '_sesgo_detectado' para el módulo causal.")


# ==============================================================================
# Curva Logit Interactiva: El Efecto Gatillo
# ==============================================================================
st.subheader("El Efecto Gatillo (Curva Logit)")
st.markdown("Aplicación de un modelo predictivo multivariado (Regresión Logística). La curva proyecta la probabilidad matemática de que ocurra un encuadre de ataque directo a partir de la consolidación de un ambiente hegemónico de negatividad en los medios.")

import statsmodels.formula.api as smf

cols_existentes_logit = [c for c in df_filtrado.columns if '_sesgo_detectado' in c]
df_analisis = df_filtrado.copy()

if cols_existentes_logit and 'prob_NEG' in df_analisis.columns and 'prob_fear' in df_analisis.columns:
    candidatos_nombres_logit = []
    for col in cols_existentes_logit:
        cand_logit = col.replace('_sesgo_detectado', '').replace('_', ' ')
        candidatos_nombres_logit.append(cand_logit)
        df_analisis[f'Ataque_{cand_logit}'] = (df_analisis[col].astype(str).str.strip().str.title() == 'Culpabilidad Activa').astype(int)

    df_analisis['fecha_semana'] = df_analisis['fecha_publicacion'].dt.to_period('W').dt.start_time

    dict_agrupacion = {'prob_fear': 'mean', 'prob_NEG': 'mean'}
    for cand_logit in candidatos_nombres_logit:
        dict_agrupacion[f'Ataque_{cand_logit}'] = 'max' 

    df_ts_logit = df_analisis.groupby(['medio_emisor', 'fecha_semana']).agg(dict_agrupacion).reset_index()

    df_ts_logit['Lag1_Miedo'] = df_ts_logit.groupby('medio_emisor')['prob_fear'].shift(1)
    df_ts_logit['Lag1_Sentimiento_Neg'] = df_ts_logit.groupby('medio_emisor')['prob_NEG'].shift(1)
    df_ts_logit = df_ts_logit.dropna(subset=['Lag1_Miedo', 'Lag1_Sentimiento_Neg'])

    df_ts_logit['Ataque_Global'] = df_ts_logit[[f'Ataque_{c}' for c in candidatos_nombres_logit]].max(axis=1)

    opciones_logit = sorted(candidatos_nombres_logit)
    default_idx_logit = opciones_logit.index("Sergio Fajardo") if "Sergio Fajardo" in opciones_logit else 0
    opcion_seleccionada = st.selectbox("Selecciona el Actor Político para evaluar Probabilidad:", opciones_logit, index=default_idx_logit, key="logit_cand")

    if not df_ts_logit.empty:
        x_neg = np.linspace(df_ts_logit['Lag1_Sentimiento_Neg'].min(), df_ts_logit['Lag1_Sentimiento_Neg'].max(), 200)

        target_col = f'Ataque_{opcion_seleccionada}'
        color_linea = '#de2d26' 
        color_puntos = 'rgba(222, 45, 38, 0.4)'

        datos_x = df_ts_logit['Lag1_Sentimiento_Neg']
        datos_y = df_ts_logit[target_col]
        fechas_texto = df_ts_logit['fecha_semana'].dt.strftime('%d-%b-%Y')
        
        if datos_y.nunique() > 1:
            try:
                modelo_cand = smf.logit(f"{target_col} ~ Lag1_Sentimiento_Neg + Lag1_Miedo", data=df_ts_logit).fit(disp=False)
                intercepto = modelo_cand.params['Intercept']
                coef_neg = modelo_cand.params['Lag1_Sentimiento_Neg']
                coef_miedo = modelo_cand.params['Lag1_Miedo']
                miedo_promedio = df_ts_logit['Lag1_Miedo'].mean()
                
                probabilidades = 1 / (1 + np.exp(-(intercepto + coef_neg * x_neg + coef_miedo * miedo_promedio)))
            except Exception:
                prob_base = datos_y.mean()
                probabilidades = np.full_like(x_neg, prob_base)
                color_linea = 'gray'
        else:
            probabilidades = np.full_like(x_neg, 0)
            color_linea = 'gray'

        import plotly.graph_objects as go
        fig5c = go.Figure()

        # A. Línea de probabilidad (La curva)
        fig5c.add_trace(go.Scatter(
            x=x_neg, 
            y=probabilidades, 
            mode='lines',
            line=dict(color=color_linea, width=4),
            name=f'Prob. Predicha ({textwrap.shorten(opcion_seleccionada, 15)})',
            hovertemplate="Nivel de Negatividad Previa: %{x:.2f}<br>Riesgo Calculado de Ataque: %{y:.1%}<extra></extra>"
        ))

        # B. Puntos reales
        fig5c.add_trace(go.Scatter(
            x=datos_x, 
            y=datos_y, 
            customdata=fechas_texto, 
            mode='markers',
            marker=dict(color=color_puntos, size=10, line=dict(width=1, color='black')),
            name=f'Semanas Reales',
            hovertemplate="<b>Semana del: %{customdata}</b><br>Negatividad de la semana anterior: %{x:.2f}<extra></extra>"
        ))

        fig5c.update_layout(
            title=dict(
                text=f"El Efecto Gatillo (Curva de Probabilidad Logística)<br><sup>Objetivo: <b>{opcion_seleccionada}</b></sup>",
                font=dict(size=20)
            ),
            plot_bgcolor='white', 
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(showgrid=True, gridcolor='whitesmoke'),
            yaxis=dict(tickformat=".0%", range=[-0.05, 1.05], showgrid=True, gridcolor='whitesmoke', tickvals=[0, 0.25, 0.5, 0.75, 1])
        )

        fig5c.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray", 
            line_width=2,
            annotation_text="Umbral del 50%", 
            annotation_position="bottom right",
            annotation_font_color="gray"
        )

        st.plotly_chart(fig5c, use_container_width=True, theme=None, config={'displayModeBar': False})
    else:
        st.info("No hay suficientes datos limpios para ejecutar la regresión logística.")
else:
    st.error("No se detectaron las columnas necesarias (prob_NEG, prob_fear o sesgo) para calcular el efecto gatillo.")
