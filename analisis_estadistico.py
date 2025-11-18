# ================================================================
# APP FERIA ESTADÍSTICA - DISEÑO FACTORIAL 2x3
# Proyecto: Etiqueta nutricional y precio vs. elección de azúcar
# Ejecutar con:  streamlit run app_feria_estadistica.py
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, f
import streamlit as st

plt.switch_backend("Agg")
sns.set(style="whitegrid")

# ================================================================
# 1. CARGA Y LIMPIEZA DEL EXCEL
# ================================================================

def cargar_datos_desde_excel(file):
    """
    Lee el Excel usando header=1 (segunda fila como encabezado),
    renombra la primera columna como 'Tratamiento', rellena hacia abajo
    y elimina filas sin respuestas.
    """
    df = pd.read_excel(file, sheet_name=0, header=1)

    # Nos quedamos con las primeras 5 columnas: Tratamiento, Edad, Sexo, Bebida, Snack
    df = df.iloc[:, :5]

    # Renombrar primera columna (RESPUESTAS ESTADÍSTICA) a 'Tratamiento'
    primera_col = df.columns[0]
    df = df.rename(columns={primera_col: "Tratamiento"})

    # Limpiar espacios en columnas de texto
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # Rellenar tratamientos
    df["Tratamiento"] = df["Tratamiento"].ffill()

    # Quitar filas sin respuestas completas
    df = df.dropna(subset=["Edad", "Sexo", "Bebida Elegida", "Snack Elegido"], how="any")

    # Convertir Edad a numérico
    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    df = df.dropna(subset=["Edad"])

    return df

# ================================================================
# 2. LIMPIEZA NOMBRES PRODUCTOS Y CÁLCULO DE AZÚCAR
# ================================================================

def limpiar_nombre(producto):
    if pd.isna(producto):
        return None

    # Quitar parte del precio → nos quedamos con el nombre antes de "—" o "-"
    p = str(producto).split("—")[0].split("-")[0].strip().lower()

    reemplazos = {
        "gaseosa": "Gaseosa",
        "jugo": "Jugo Hit",
        "té": "Té",
        "te ": "Té",
        "agua": "Agua",
        "ponqué": "Ponqué",
        "ponque": "Ponqué",
        "galletas": "Galletas",
        "barra": "Barra de cereal",
        "manzana": "Manzana"
    }

    for key, val in reemplazos.items():
        if key in p:
            return val

    return producto  # por si aparece algo extraño

AZUCAR_PRODUCTOS = {
    "Gaseosa": 35,
    "Jugo Hit": 28,
    "Té": 15,
    "Agua": 0,
    "Ponqué": 22,
    "Galletas": 12,
    "Barra de cereal": 8,
    "Manzana": 0
}

def calcular_azucar(df, umbral_saludable=15):
    df["Bebida"] = df["Bebida Elegida"].apply(limpiar_nombre)
    df["Snack"]  = df["Snack Elegido"].apply(limpiar_nombre)

    df["AzucarBebida"] = df["Bebida"].map(AZUCAR_PRODUCTOS)
    df["AzucarSnack"]  = df["Snack"].map(AZUCAR_PRODUCTOS)

    df["AzucarTotal"] = df["AzucarBebida"] + df["AzucarSnack"]
    df["Saludable"]   = (df["AzucarTotal"] <= umbral_saludable).astype(int)

    return df

# ================================================================
# 3. ASIGNAR FACTORES A (ETIQUETA) Y B (PRECIO)
# ================================================================

def asignar_factores(df):
    """
    A: Etiqueta
       - SIN etiqueta → Tratamientos 1,2,3
       - CON etiqueta → Tratamientos 4,5,6
    B: Precio
       - Igual      → Tratamientos 1 y 4
       - Descuento  → Tratamientos 2 y 5
       - Recargo    → Tratamientos 3 y 6
    """
    df["TratamientoNum"] = df["Tratamiento"].astype(str).str.extract(r"(\d+)").astype(int)

    # Factor A: Etiqueta
    df["Etiqueta"] = df["TratamientoNum"].apply(
        lambda x: "Sin etiqueta" if x in [1, 2, 3] else "Con etiqueta"
    )

    # Factor B: Precio
    precio_map = {
        1: "Igual",
        2: "Descuento",
        3: "Recargo",
        4: "Igual",
        5: "Descuento",
        6: "Recargo",
    }
    df["Precio"] = df["TratamientoNum"].map(precio_map)

    return df

# ================================================================
# 4. ESTIMACIONES E INTERVALOS DE CONFIANZA
# ================================================================

def ic_media(x, alpha=0.05):
    """
    Intervalo de confianza para la media con varianza poblacional desconocida (t-Student)
    """
    x = np.array(x)
    n = len(x)
    media = x.mean()
    s = x.std(ddof=1)
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    margen = t_crit * s / np.sqrt(n)
    return media, media - margen, media + margen

def ic_proporcion(p_hat, n, alpha=0.05):
    """
    Intervalo de confianza normal aproximado para una proporción.
    """
    z_crit = norm.ppf(1 - alpha/2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margen = z_crit * se
    return p_hat, p_hat - margen, p_hat + margen

# ================================================================
# 5. PRUEBA DE PROPORCIONES Y ANOVA FACTORIAL
# ================================================================

def prueba_proporciones(p1, p2, n1, n2):
    p_pool = (p1*n1 + p2*n2) / (n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (p1 - p2)/se
    p_value = 2*(1 - norm.cdf(abs(z)))
    return z, p_value

def anova_factorial(df, factorA, factorB, y):
    A = df[factorA].unique()
    B = df[factorB].unique()

    n = df.groupby([factorA, factorB]).size().iloc[0]
    N = len(df)
    media_global = df[y].mean()

    # Sumas de cuadrados
    SSA = sum([
        n * (df[df[factorA]==a][y].mean() - media_global)**2
        for a in A
    ])

    SSB = sum([
        n * (df[df[factorB]==b][y].mean() - media_global)**2
        for b in B
    ])

    SSAB = 0
    for a in A:
        for b in B:
            media_ab = df[(df[factorA]==a)&(df[factorB]==b)][y].mean()
            media_a  = df[df[factorA]==a][y].mean()
            media_b  = df[df[factorB]==b][y].mean()
            SSAB += n*(media_ab - media_a - media_b + media_global)**2

    SSE = sum((df[y] - df.groupby([factorA,factorB])[y].transform("mean"))**2)
    SST = SSA + SSB + SSAB + SSE

    # Grados de libertad
    a = len(A)
    b = len(B)
    dfA  = a-1
    dfB  = b-1
    dfAB = dfA*dfB
    dfE  = N - a*b

    # Cuadrados medios
    MSA  = SSA/dfA
    MSB  = SSB/dfB
    MSAB = SSAB/dfAB
    MSE  = SSE/dfE

    # Estadísticos F y p-values
    FA  = MSA/MSE
    FB  = MSB/MSE
    FAB = MSAB/MSE

    pA  = 1 - f.cdf(FA, dfA, dfE)
    pB  = 1 - f.cdf(FB, dfB, dfE)
    pAB = 1 - f.cdf(FAB, dfAB, dfE)

    tabla = pd.DataFrame({
        "Fuente": ["Etiqueta (A)", "Precio (B)", "Interacción AB", "Error", "Total"],
        "SS": [SSA, SSB, SSAB, SSE, SST],
        "df": [dfA, dfB, dfAB, dfE, N-1],
        "MS": [MSA, MSB, MSAB, MSE, ""],
        "F": [FA, FB, FAB, "", ""],
        "p-value": [pA, pB, pAB, "", ""]
    })

    return tabla, (FA, FB, FAB, dfA, dfB, dfAB, dfE)

# ================================================================
# 6. GRÁFICOS
# ================================================================

def grafico_bar_saludable(df):
    fig, ax = plt.subplots(figsize=(8,4))
    prop = df.groupby("Tratamiento")["Saludable"].mean().reset_index()
    sns.barplot(data=prop, x="Tratamiento", y="Saludable", ax=ax)
    ax.set_ylim(0,1)
    ax.set_title("Proporción de elecciones saludables por tratamiento")
    ax.set_ylabel("Proporción saludable")
    return fig

def grafico_bar_azucar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    media = df.groupby("Tratamiento")["AzucarTotal"].mean().reset_index()
    sns.barplot(data=media, x="Tratamiento", y="AzucarTotal", ax=ax)
    ax.set_title("Azúcar promedio por tratamiento (g)")
    ax.set_ylabel("Azúcar total (g)")
    return fig

def grafico_box_azucar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df, x="Tratamiento", y="AzucarTotal", ax=ax)
    ax.set_title("Distribución de azúcar por tratamiento")
    ax.set_ylabel("Azúcar total (g)")
    return fig

def grafico_interaccion(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.pointplot(data=df, x="Precio", y="AzucarTotal", hue="Etiqueta",
                  dodge=True, ax=ax)
    ax.set_title("Interacción Etiqueta × Precio en azúcar total")
    ax.set_ylabel("Azúcar total (g)")
    return fig

# ================================================================
# 7. INTERFAZ STREAMLIT
# ================================================================

def main():
    st.title("Feria Estadística – Diseño factorial 2×3")
    st.write("""
    **Proyecto:** Influencia del etiquetado nutricional y del precio en la elección
    de productos con distinto contenido de azúcar en estudiantes de 9.º a 11.º.
    """)

    uploaded_file = st.file_uploader("Sube el archivo de respuestas (.xlsx)", type=["xlsx"])

    umbral = st.sidebar.slider(
        "Umbral de azúcar para considerar 'saludable' (g)",
        min_value=0, max_value=50, value=15, step=1
    )

    alpha = st.sidebar.slider(
        "Nivel de significancia α",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )

    if uploaded_file is None:
        st.info("Por favor sube el archivo **RespuestasEncuestaEstadistica.xlsx**.")
        return

    # ----------------- Procesamiento de datos -----------------
    df = cargar_datos_desde_excel(uploaded_file)
    df = asignar_factores(df)
    df = calcular_azucar(df, umbral_saludable=umbral)

    st.subheader("Datos procesados (primeras filas)")
    st.dataframe(df.head(15))

    # ----------------- Descriptiva -----------------
    st.subheader("1. Análisis descriptivo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Media, mediana y cuartiles de azúcar total (g)**")
        st.write(df["AzucarTotal"].describe()[["mean","50%","25%","75%"]])
    with col2:
        st.markdown("**Proporción de elecciones saludables por tratamiento**")
        st.write(df.groupby("Tratamiento")["Saludable"].mean())

    # ----------------- Estimaciones e IC -----------------
    st.subheader("2. Estimaciones puntuales e intervalos de confianza")

    # Media de azúcar total
    media, li_m, ls_m = ic_media(df["AzucarTotal"], alpha=alpha)
    st.markdown("**2.1. Media poblacional de azúcar total (g)**")
    st.write(f"Estimación puntual:  x̄ = {media:.2f} g")
    st.write(f"IC {int((1-alpha)*100)}%: ({li_m:.2f} g ; {ls_m:.2f} g)")
    st.markdown(
        f"_Interpretación_: Con un nivel de confianza del {int((1-alpha)*100)} %, "
        f"podemos afirmar que la **media real de azúcar total** consumida por los estudiantes "
        f"se encuentra entre **{li_m:.2f} g** y **{ls_m:.2f} g**."
    )

    # Proporción de elecciones saludables
    p_hat = df["Saludable"].mean()
    n_total = len(df)
    p_est, li_p, ls_p = ic_proporcion(p_hat, n_total, alpha=alpha)
    st.markdown("**2.2. Proporción poblacional de elecciones saludables**")
    st.write(f"Estimación puntual:  p̂ = {p_est:.3f}")
    st.write(f"IC {int((1-alpha)*100)}%: ({li_p:.3f} ; {ls_p:.3f})")
    st.markdown(
        f"_Interpretación_: Con un nivel de confianza del {int((1-alpha)*100)} %, "
        f"la **proporción real de estudiantes que eligen una combinación saludable** "
        f"se encuentra entre **{li_p:.3f}** y **{ls_p:.3f}**."
    )

    # ----------------- Prueba de hipótesis (proporciones) -----------------
    st.subheader("3. Prueba de hipótesis: efecto de la etiqueta en la proporción saludable")

    p1 = df[df["Etiqueta"]=="Con etiqueta"]["Saludable"].mean()
    p2 = df[df["Etiqueta"]=="Sin etiqueta"]["Saludable"].mean()
    n1 = len(df[df["Etiqueta"]=="Con etiqueta"])
    n2 = len(df[df["Etiqueta"]=="Sin etiqueta"])
    z, p_value = prueba_proporciones(p1, p2, n1, n2)

    st.write(f"p₁ = proporción saludable **con etiqueta** = {p1:.3f} (n₁ = {n1})")
    st.write(f"p₂ = proporción saludable **sin etiqueta** = {p2:.3f} (n₂ = {n2})")
    st.write(f"Estadístico Z = {z:.3f}")
    st.write(f"Valor-p = {p_value:.4f}")

    st.markdown("""
    **Hipótesis planteadas**

    - H₀: p₁ = p₂ → la etiqueta **no cambia** la proporción de elecciones saludables.  
    - H₁: p₁ ≠ p₂ → la etiqueta **sí cambia** la proporción de elecciones saludables.
    """)

    if p_value < alpha:
        st.success(
            f"Como p-value = {p_value:.4f} < α = {alpha:.2f}, se **rechaza H₀**. "
            "Concluimos que el etiquetado nutricional tiene un efecto estadísticamente "
            "significativo en la proporción de elecciones saludables."
        )
    else:
        st.info(
            f"Como p-value = {p_value:.4f} ≥ α = {alpha:.2f}, **no se rechaza H₀**. "
            "Con los datos de esta muestra, no hay evidencia suficiente para afirmar que la etiqueta "
            "modifique la proporción de estudiantes que eligen opciones saludables."
        )

    # ----------------- ANOVA factorial 2×3 -----------------
    st.subheader("4. ANOVA factorial 2×3 (Etiqueta × Precio) sobre azúcar total")

    tabla_anova, anova_info = anova_factorial(df, "Etiqueta", "Precio", "AzucarTotal")
    st.dataframe(tabla_anova)

    FA, FB, FAB, dfA, dfB, dfAB, dfE = anova_info
    pA = tabla_anova.loc[0, "p-value"]
    pB = tabla_anova.loc[1, "p-value"]
    pAB = tabla_anova.loc[2, "p-value"]

    st.markdown("""
    **Hipótesis del ANOVA**

    - Para Etiqueta (A):  
      H₀: las medias de azúcar son iguales con y sin etiqueta.  
      H₁: al menos una de las medias difiere.

    - Para Precio (B):  
      H₀: las medias de azúcar son iguales entre las estrategias de precio (igual, descuento, recargo).  
      H₁: al menos una de las medias difiere.

    - Para la Interacción A×B:  
      H₀: no hay interacción entre etiqueta y precio.  
      H₁: sí hay interacción; el efecto de uno depende del otro.
    """)

    # Interpretación de cada factor
    def interpreta_factor(nombre, p_val):
        if p_val < alpha:
            st.success(
                f"Para **{nombre}**: p-value = {p_val:.4f} < α = {alpha:.2f} ⇒ "
                f"se **rechaza H₀**. Hay evidencia de que {nombre.lower()} "
                "tiene un efecto significativo en el azúcar total elegido."
            )
        else:
            st.info(
                f"Para **{nombre}**: p-value = {p_val:.4f} ≥ α = {alpha:.2f} ⇒ "
                f"**no se rechaza H₀**. Con esta muestra no hay evidencia "
                f"suficiente de que {nombre.lower()} afecte el azúcar total elegido."
            )

    interpreta_factor("la ETIQUETA", pA)
    interpreta_factor("el PRECIO", pB)
    interpreta_factor("la INTERACCIÓN Etiqueta × Precio", pAB)

    st.markdown(
        "_En términos del problema_, el ANOVA nos dice si el etiquetado, "
        "las estrategias de precio o la combinación de ambos cambian de manera "
        "significativa la cantidad de azúcar que terminan escogiendo los estudiantes."
    )

    # ----------------- Gráficos -----------------
    st.subheader("5. Gráficos")

    st.markdown("**5.1 Proporción de elecciones saludables por tratamiento**")
    st.pyplot(grafico_bar_saludable(df))

    st.markdown("**5.2 Azúcar promedio por tratamiento**")
    st.pyplot(grafico_bar_azucar(df))

    st.markdown("**5.3 Distribución de azúcar por tratamiento (boxplot)**")
    st.pyplot(grafico_box_azucar(df))

    st.markdown("**5.4 Interacción Etiqueta × Precio en el azúcar total**")
    st.pyplot(grafico_interaccion(df))

    # ----------------- Conclusión resumida -----------------
    st.subheader("6. Conclusión resumida (para el informe)")

    st.markdown(f"""
    - El **promedio de azúcar total** consumida por combinación bebida+snack se estima en
      aproximadamente **{media:.2f} g**, con un intervalo de confianza del {int((1-alpha)*100)} %
      entre **{li_m:.2f} g** y **{ls_m:.2f} g**.
    - La **proporción global de elecciones saludables** (≤ {umbral} g de azúcar) es aproximadamente
      **{p_est:.3f}**, con IC del {int((1-alpha)*100)} % entre **{li_p:.3f}** y **{ls_p:.3f}**.
    - En la **prueba de proporciones** para el efecto de la etiqueta, el valor-p fue **{p_value:.4f}**.
      {"Esto indica un efecto significativo del etiquetado." if p_value < alpha else "Esto indica que, con esta muestra, no se detecta un efecto significativo del etiquetado."}
    - En el **ANOVA factorial 2×3**, se evaluó el efecto de la **etiqueta**, del **precio** y de su
      **interacción** sobre el azúcar total elegido por los estudiantes. Los valores-p indican si cada
      factor tiene o no un efecto estadísticamente significativo en el comportamiento de consumo.
    """)

# ================================================================
# 8. MAIN
# ================================================================

if __name__ == "__main__":
    main()
