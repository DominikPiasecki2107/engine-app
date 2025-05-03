# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import f_oneway, shapiro, levene, kurtosis, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io # Do obsługi wgranych plików w pamięci

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Kompleksowa analiza silników")
st.title("Kompleksowa analiza silników - Wersja Webowa")

# --- Definicje (takie same jak w oryginale) ---
STAT_DEFINITIONS = {
    "count": ("Liczba pomiarów", "Liczba dostępnych obserwacji w danej kolumnie danych"),
    "mean": ("Średnia", "Wartość przeciętna – suma wszystkich wartości podzielona przez ich liczbę"),
    "median": ("Mediana", "Wartość środkowa – dzieli uporządkowany zbiór danych na dwie równe części"),
    "std": ("Odchylenie standardowe", "Miara rozrzutu – pokazuje, jak bardzo wartości odbiegają średnio od średniej"),
    "min": ("Minimum", "Najniższa obserwowana wartość w zbiorze danych"),
    "max": ("Maksimum", "Najwyższa obserwowana wartość w zbiorze danych"),
    "skew": ("Skosność", "Miara asymetrii rozkładu: dodatnia oznacza długi ogon z prawej strony osi wartości (większość danych jest po lewej), ujemna – ogon z lewej strony (większość danych po prawej stronie osi wartości)"),
    "kurtosis": ("Kurtoza", "Miara „spiczastości” rozkładu: pokazuje, czy dane mają tendencję do występowania dużej liczby skrajnych wartości"),
    "percentile_95": ("95 percentyl", "Wartość graniczna, poniżej której znajduje się 95% wszystkich obserwacji"),
    "iqr": ("Rozstęp międzykwartylowy", "Zakres między wartością 25. a 75. percentyla – obejmuje środkowe 50% danych i jest odporny na wartości skrajne"),
    "var": ("Wariancja", "Miara zmienności – średnia kwadratów różnic między wartościami a średnią; pokazuje ogólne rozproszenie danych")
}

TAB_DEFINITIONS = {
    "raw_data": {
        "title": "Surowe dane",
        "description": "Podgląd pierwszych wierszy załadowanych danych."
    },
    "advanced": {
        "title": "Analiza statystyczna",
        "description": "Zakładka zawiera szczegółowe statystyki opisowe dla wybranych zmiennych z obu silników.",
        "terms": STAT_DEFINITIONS,
        "plots": {
            "boxplot": ("Wykres pudełkowy", "Pokazuje rozkład danych, medianę, kwartyle i wartości odstające"),
            "histogram": ("Histogram", "Pokazuje rozkład częstości wartości w przedziałach")
        }
    },
    "correlation": {
        "title": "Analiza korelacji",
        "description": "Sprawdzenie, czy i jak mocno dwie kolumny danych są ze sobą powiązane.",
        "terms": {
            "spearman": ("Korelacja Spearmana",
                        "Pokazuje, czy gdy jedna wartość rośnie, to druga też zwykle rośnie lub maleje."),
            "heatmap": ("Mapa cieplna",
                        "Tabela krzyżowa pokazująca powiązania między kolumnami. Watość blisko 1 oznacza, że wartości rosną razem. Blisko -1 – że jedna rośnie, a druga maleje."),
            "strong_corr": ("Silna korelacja",
                            "Jeśli liczba jest większa niż 0.7 lub mniejsza niż -0.7, to znaczy, że dwie kolumny są mocno powiązane."),
        }
    },
    "regression": {
        "title": "Analiza regresji",
        "description": "Ocena, w jakim stopniu jedna lub więcej zmiennych może przewidywać wartość innej zmiennej. Regresja pozwala określić relacje i siłę wpływu między zmiennymi.",
        "terms": {
            "r_squared": (
                "R-kwadrat",
                "Miara dopasowania modelu regresji. Pokazuje, jaka część zmienności zmiennej zależnej (Y) może być wyjaśniona przez zmienne niezależne (X). Wartość R² zawiera się w przedziale od 0 do 1. Im bliżej 1, tym lepsze dopasowanie modelu do danych."
            ),
            "p_value": (
                "Wartość p",
                "Miara istotności statystycznej wyników. Informuje, jakie jest prawdopodobieństwo uzyskania podobnych wyników, jeśli nie istnieje faktyczny związek pomiędzy badanymi zmiennymi. Niska wartość p (np. < 0.05) świadczy o tym, że zależność jest istotna statystycznie."
            ),
            "coefficient": (
                "Współczynnik regresji",
                "Liczbowa wartość wskazująca kierunek i siłę wpływu danej zmiennej niezależnej na zmienną zależną. Dodatni współczynnik oznacza, że wraz ze wzrostem X rośnie Y, a ujemny – że Y maleje. Wielkość współczynnika mówi, o ile zmienia się Y, gdy X rośnie o jednostkę."
            ),
            "residuals": (
                "Reszty",
                "Różnice między rzeczywistymi wartościami zmiennej zależnej a wartościami przewidywanymi przez model regresji. Analiza reszt pozwala ocenić, czy model jest dobrze dopasowany i czy spełnia założenia (np. normalność, losowość)."
            ),
            "h0": (
                "Hipoteza zerowa (H0)",
                "Początkowe założenie w testach statystycznych mówiące, że między badanymi zmiennymi nie istnieje istotna zależność. W analizie regresji H0 zakłada, że współczynnik regresji jest równy zeru. Jeśli test pozwoli odrzucić H0 (np. przy p < 0.05), uznaje się, że zmienne są powiązane."
            )
        },
        "plots": {
            "residual_plot": (
                "Wykres reszt",
                "Wizualizacja błędów predykcji modelu – czyli różnic między przewidywanymi a rzeczywistymi wartościami. Pomaga ocenić, czy model spełnia założenia statystyczne (np. losowy rozkład błędów, brak wzorców)."
            ),
            "qq_plot": (
                "Wykres Q–Q",
                "Wykres porównujący rozkład reszt modelu z idealnym rozkładem normalnym. Służy do oceny, czy błędy modelu są rozłożone symetrycznie i zgodnie z teorią – co jest kluczowym założeniem dla regresji liniowej."
            )
        }
    },
    "diagnostics": {
        "title": "Diagnostyka",
        "description": "Sprawdzenie, czy dane spełniają warunki potrzebne do poprawnej analizy.",
        "terms": {
            "t_test":("Test t-Studenta (Welcha)",
                        "Metoda statystyczna służąca do porównania średnich między dwoma grupami. Wersja Welcha nie zakłada równości wariancji. Wynik testu pokazuje, czy różnica między średnimi jest istotna statystycznie."),
            "shapiro": ("Test Shapiro-Wilka",
                        "Statystyczna metoda sprawdzająca, czy dane mają rozkład normalny, czyli taki, który jest symetryczny i skoncentrowany wokół średniej. Test ten weryfikuje, czy dane spełniają jedno z podstawowych założeń wielu analiz statystycznych.”."),
            "levene": ("Test Levene'a",
                    "Służy do oceny, czy zmienność (rozrzut) wartości w różnych grupach jest podobna. Jeśli wyniki testu są istotne, oznacza to, że jedna grupa ma większą zmienność niż druga, co może wpływać na dalsze analizy."),
            "anova": ("ANOVA",
                    "Procedura statystyczna pozwalająca sprawdzić, czy średnie wartości danej cechy w różnych grupach różnią się istotnie. Nie mówi, które grupy się różnią – tylko że przynajmniej jedna różni się od pozostałych."),
            "p_value": ("Wartość p",
                        "Jest to liczba, która pokazuje, jak bardzo uzyskany wynik odbiega od sytuacji, w której nie ma żadnej różnicy między grupami. Im mniejsza wartość p (np. < 0.05), tym większa pewność, że wynik nie jest przypadkowy i można go uznać za statystycznie istotny."),
            "h0": ("Hipoteza zerowa (H0)",
                "To podstawowe założenie każdego testu statystycznego – że nie ma różnicy między grupami, które porównujemy. Dopiero test wskazuje, czy mamy podstawy, by to założenie odrzucić i uznać, że różnica jednak istnieje.")
        }
    },
    "custom_plot": {
        "title": "Niestandardowy wykres",
        "description": "Twórz własne wykresy, wybierając zmienne dla osi X i Y.",
        "terms": {
            "reference_line": ("Linia wzorcowa",
                                "Pozioma lub pionowa linia wskazująca wartość referencyjną"),
            "scatter": ("Wykres punktowy",
                        "Pokazuje zależność między dwiema zmiennymi ilościowymi"),
            "grouping": ("Grupowanie",
                        "Rozróżnianie danych z różnych źródeł za pomocą kolorów/markerów")
        }
    },
    "power_rpm": {
        "title": "Moc vs Obroty",
        "description": "Analiza zależności mocy od obrotów silnika wraz z porównaniem do krzywej wzorcowej i analizą bezpieczeństwa.",
         "terms": {
             "reference_curve": ("Krzywa wzorcowa", "Teoretyczna lub oczekiwana charakterystyka mocy silnika w zależności od obrotów."),
             "rpm_sectors": ("Sektory obrotów", "Podział zakresu obrotów na części (np. 3-4k, 4-5k, 5-6k) w celu analizy charakterystyki w różnych reżimach pracy."),
             "safety_analysis": ("Analiza bezpieczeństwa", "Ocena stabilności i odchylenia od wzorca mocy silnika w poszczególnych sektorach obrotów.")
         }
    }
}

# --- Inicjalizacja stanu sesji (przechowuje dane między interakcjami) ---
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None
if 'df_combined' not in st.session_state:
    st.session_state.df_combined = None
if 'df1_name' not in st.session_state:
    st.session_state.df1_name = "Silnik 1"
if 'df2_name' not in st.session_state:
    st.session_state.df2_name = "Silnik 2"
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'reference_lines' not in st.session_state:
    st.session_state.reference_lines = [] # Lista krotek (x_val, y_val)


# --- Funkcje pomocnicze ---
def show_tab_info(tab_key):
    """Wyświetla rozwijane sekcje z informacjami o zakładce."""
    tab_info = TAB_DEFINITIONS.get(tab_key, {})
    if not tab_info:
        return

    with st.expander("ℹ️ Informacje o zakładce", expanded=False):
        st.markdown(f"**{tab_info.get('title', '')}**")
        st.caption(tab_info.get('description', ''))

        if "terms" in tab_info and tab_info["terms"]:
            st.markdown("**Terminy:**")
            for term, (name, definition) in tab_info["terms"].items():
                 st.markdown(f"- **{name}:** {definition}")

        if "plots" in tab_info and tab_info["plots"]:
             st.markdown("**Wykresy:**")
             for plot, (name, definition) in tab_info["plots"].items():
                 st.markdown(f"- **{name}:** {definition}")

def prepare_data(df1, df2, name1, name2):
    """Łączy DataFrame'y, dodaje kolumnę źródła i identyfikuje kolumny numeryczne."""
    if df1 is None or df2 is None:
        return None, []

    try:
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy["Źródło"] = name1
        df2_copy["Źródło"] = name2
        df_combined = pd.concat([df1_copy, df2_copy], ignore_index=True)

        numeric_columns = df_combined.select_dtypes(include=np.number).columns.tolist()
        # Opcjonalnie usuń kolumny, które nie powinny być traktowane jako numeryczne do analizy
        # np. numeric_columns = [col for col in numeric_columns if col not in ['ID', 'Rok']]

        if 'Data' in df_combined.columns:
            try: # Dodatkowa obsługa błędów konwersji daty
                 df_combined['Data'] = pd.to_datetime(df_combined['Data'], errors='coerce')
            except Exception as e:
                 st.warning(f"Problem z konwersją kolumny 'Data': {e}")


        return df_combined, numeric_columns
    except Exception as e:
        st.error(f"Błąd podczas przygotowywania danych: {e}")
        return None, []

def clear_data():
    """Resetuje stan sesji."""
    st.session_state.df1 = None
    st.session_state.df2 = None
    st.session_state.df_combined = None
    st.session_state.df1_name = "Silnik 1"
    st.session_state.df2_name = "Silnik 2"
    st.session_state.numeric_columns = []
    st.session_state.reference_lines = []
    # Reset kluczy specyficznych dla widgetów, aby odświeżyć ich stan
    keys_to_reset = [key for key in st.session_state if key.startswith("sel_") or key.startswith("inp_")]
    for key in keys_to_reset:
        del st.session_state[key]
    st.success("Dane zostały wyczyszczone.")


# --- Funkcje dla Krzywych Wzorcowych i Analizy Bezpieczeństwa (takie same jak w oryginale) ---
def get_reference_curve_19():
    obroty = np.linspace(3000, 6000, 300)
    peak = 130  # szczyt mocy
    start = 85  # moc przy 3000 obr
    # funkcja wzrostu (sigmoid + opadanie po 5000)
    moc = []
    for obr in obroty:
        if obr <= 5000:
            m = start + (peak - start) * (1 - np.exp(-0.0025 * (obr - 3000)))
        else:
            m = peak - 0.01 * (obr - 5000)  # delikatne opadanie
        moc.append(m)
    return obroty, np.array(moc)

def get_reference_curve_20():
    obroty = np.linspace(3000, 6000, 300)
    peak = 140
    start = 90
    # funkcja wzrostu (sigmoid + opadanie po 4800)
    moc = []
    for obr in obroty:
        if obr <= 4800:
            m = start + (peak - start) * (1 - np.exp(-0.003 * (obr - 3000)))
        else:
            m = peak - 0.012 * (obr - 4800)  # łagodne zejście
        moc.append(m)
    return obroty, np.array(moc)

def analyze_engine_safety(engine_name, df_combined, reference_type):
    """Analizuje bezpieczeństwo silnika porównując do krzywej wzorcowej."""
    if df_combined is None or 'Moc_KM' not in df_combined.columns or 'Moc_obr' not in df_combined.columns:
         st.warning("Brak wymaganych kolumn ('Moc_KM', 'Moc_obr') do analizy bezpieczeństwa.")
         return []

    df = df_combined[df_combined["Źródło"] == engine_name]
    if df.empty:
        st.warning(f"Brak danych dla silnika: {engine_name}")
        return []

    # Sprawdzenie typów danych i próba konwersji
    try:
        obroty = df["Moc_obr"].astype(float).values
        moc = df["Moc_KM"].astype(float).values
    except ValueError:
        st.error("Kolumny 'Moc_obr' lub 'Moc_KM' zawierają wartości, których nie można przekonwertować na liczby.")
        return []
    except KeyError:
         st.error("Brak kolumn 'Moc_obr' lub 'Moc_KM'.")
         return []


    df_real = pd.DataFrame({"Moc_obr": obroty, "Moc_KM": moc})

    if reference_type == "1.9":
        obroty_ref, moc_ref = get_reference_curve_19()
    else: # Zakładamy, że druga opcja to "2.0"
        obroty_ref, moc_ref = get_reference_curve_20()

    sectors = [(3000, 4000), (4000, 5000), (5000, 6000)]
    results = []

    for start, end in sectors:
        mask = (df_real["Moc_obr"] >= start) & (df_real["Moc_obr"] < end)
        sector_data = df_real[mask]

        if len(sector_data) == 0:
            results.append({
                "Sektor": f"{start}-{end} obr/min",
                "Śr. moc": np.nan, "Odch. std": np.nan, "Rozstęp": np.nan,
                "Śr. różnica do wzorca": np.nan, "Wskaźnik bezpieczeństwa": np.nan
            })
            continue # Przejdź do następnego sektora jeśli brak danych

        moc_real = sector_data["Moc_KM"].values
        obr_real = sector_data["Moc_obr"].values

        # Upewnij się, że obroty ref pokrywają zakres obr_real, inaczej interpolacja da NaN
        if obr_real.min() < obroty_ref.min() or obr_real.max() > obroty_ref.max():
             st.warning(f"Zakres obrotów w danych ({obr_real.min()}-{obr_real.max()}) wykracza poza zakres krzywej referencyjnej ({obroty_ref.min()}-{obroty_ref.max()}) dla sektora {start}-{end}. Wyniki mogą być niemiarodajne.")
             # Można rozważyć ekstrapolację lub ograniczenie analizy do wspólnego zakresu
             # Tutaj po prostu interpolujemy, co może dać NaN na krańcach
             moc_ref_interp = np.interp(obr_real, obroty_ref, moc_ref, left=np.nan, right=np.nan)
        else:
            moc_ref_interp = np.interp(obr_real, obroty_ref, moc_ref)


        # Usuń NaN powstałe z interpolacji jeśli zakresy się nie pokrywały
        valid_indices = ~np.isnan(moc_ref_interp)
        if not np.any(valid_indices): # Jeśli wszystkie są NaN
            diff = np.array([np.nan]) # Ustaw różnicę na NaN
        else:
             diff = moc_real[valid_indices] - moc_ref_interp[valid_indices]

        srednia = np.mean(moc_real)
        std = np.std(moc_real)
        rozstep = np.max(moc_real) - np.min(moc_real) if len(moc_real) > 0 else np.nan
        sr_roznica = np.mean(np.abs(diff)) if len(diff) > 0 and not np.all(np.isnan(diff)) else np.nan # Średnia różnica tylko jeśli są prawidłowe wartości

        # Oblicz wskaźnik tylko jeśli mamy wszystkie potrzebne dane
        if np.isnan(sr_roznica) or np.isnan(std) or np.isnan(rozstep):
            ocena = np.nan
        else:
            ocena = 0.4 * sr_roznica + 0.4 * std + 0.2 * rozstep

        results.append({
            "Sektor": f"{start}-{end} obr/min",
            "Śr. moc": round(srednia, 2) if not np.isnan(srednia) else np.nan,
            "Odch. std": round(std, 2) if not np.isnan(std) else np.nan,
            "Rozstęp": round(rozstep, 2) if not np.isnan(rozstep) else np.nan,
            "Śr. różnica do wzorca": round(sr_roznica, 2) if not np.isnan(sr_roznica) else np.nan,
            "Wskaźnik bezpieczeństwa": round(ocena, 2) if not np.isnan(ocena) else np.nan
        })

    # Sortuj tylko jeśli wskaźniki nie są NaN
    valid_results = [r for r in results if not np.isnan(r["Wskaźnik bezpieczeństwa"])]
    nan_results = [r for r in results if np.isnan(r["Wskaźnik bezpieczeństwa"])]

    sorted_valid_results = sorted(valid_results, key=lambda x: x["Wskaźnik bezpieczeństwa"])

    return sorted_valid_results + nan_results # Połącz posortowane wyniki z tymi, których nie dało się ocenić


# --- Panel Boczny (Sidebar) do wczytywania danych ---
st.sidebar.header("Wczytywanie danych")

uploaded_file1 = st.sidebar.file_uploader("Wybierz plik Excel dla Silnika 1", type=["xlsx"], key="file1")
name1 = st.sidebar.text_input("Nazwa dla Silnika 1", value=st.session_state.df1_name, key="name1")

uploaded_file2 = st.sidebar.file_uploader("Wybierz plik Excel dla Silnika 2", type=["xlsx"], key="file2")
name2 = st.sidebar.text_input("Nazwa dla Silnika 2", value=st.session_state.df2_name, key="name2")

process_button = st.sidebar.button("Przetwórz pliki", key="process")
clear_button = st.sidebar.button("Wyczyść dane", key="clear")

# Przycisk "O programie" w panelu bocznym
st.sidebar.header("Pomoc")
with st.sidebar.expander("O programie"):
    st.markdown("""
    **Aplikacja do kompleksowej analizy danych silników (Wersja Web)**

    Wersja: 1.1 (Web)
    Autor: [Dominik Piasecki] / Konwersja: AI Gemini

    Funkcjonalności:
    - Wczytywanie danych z dwóch plików Excel.
    - Podgląd surowych danych.
    - Szczegółowa analiza statystyczna.
    - Analiza korelacji (heatmapa Spearmana).
    - Analiza regresji liniowej (OLS) z diagnostyką.
    - Testy diagnostyczne (normalność, równość wariancji, ANOVA, t-test).
    - Tworzenie niestandardowych wykresów z liniami wzorcowymi.
    - Analiza zależności mocy od obrotów z krzywą wzorcową i oceną bezpieczeństwa.

    © 2025 Wszelkie prawa zastrzeżone (dla oryginalnej aplikacji).
    """)

if clear_button:
    clear_data()
    st.rerun() # Wymuś przeładowanie strony, aby wyczyścić stan

# Logika przetwarzania danych po naciśnięciu przycisku
if process_button:
    if uploaded_file1 and uploaded_file2 and name1 and name2:
        try:
            df1 = pd.read_excel(uploaded_file1)
            df2 = pd.read_excel(uploaded_file2)
            st.session_state.df1 = df1
            st.session_state.df2 = df2
            st.session_state.df1_name = name1
            st.session_state.df2_name = name2
            st.session_state.df_combined, st.session_state.numeric_columns = prepare_data(
                st.session_state.df1, st.session_state.df2,
                st.session_state.df1_name, st.session_state.df2_name
            )
            if st.session_state.df_combined is not None:
                 st.sidebar.success(f"Wczytano '{name1}' ({len(df1)} wierszy) i '{name2}' ({len(df2)} wierszy).")
                 st.rerun() # Przeładuj, aby odświeżyć zakładki z nowymi danymi
            else:
                 st.sidebar.error("Nie udało się połączyć danych.")

        except Exception as e:
            st.sidebar.error(f"Błąd wczytywania lub przetwarzania plików: {e}")
    else:
        st.sidebar.warning("Proszę wgrać oba pliki i podać obie nazwy.")

# --- Główna część aplikacji (Zakładki) ---
if st.session_state.df_combined is not None and not st.session_state.df_combined.empty:

    tab_keys = ["raw_data", "advanced", "correlation", "regression", "diagnostics", "custom_plot", "power_rpm"]
    tab_titles = [TAB_DEFINITIONS.get(k, {}).get("title", k.replace("_", " ").title()) for k in tab_keys]

    tabs = st.tabs(tab_titles)

    # --- Zakładka: Surowe dane ---
    with tabs[0]:
        key = "raw_data"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)
        st.subheader(f"Podgląd danych: {st.session_state.df1_name}")
        st.dataframe(st.session_state.df1.head(10))
        st.subheader(f"Podgląd danych: {st.session_state.df2_name}")
        st.dataframe(st.session_state.df2.head(10))

    # --- Zakładka: Analiza statystyczna ---
    with tabs[1]:
        key = "advanced"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns:
            st.warning("Brak kolumn numerycznych do analizy.")
        else:
            # Użycie unikalnego klucza dla selectbox
            selected_column_adv = st.selectbox(
                "Wybierz kolumnę do analizy:",
                st.session_state.numeric_columns,
                key="sel_adv_col" # Unikalny klucz
            )

            if selected_column_adv:
                st.subheader(f"Statystyki opisowe dla: {selected_column_adv}")
                try:
                    stats = st.session_state.df_combined.groupby("Źródło")[selected_column_adv].agg([
                        "count", "min", "max", "mean", "median", "std", "var",
                        "skew", lambda x: kurtosis(x, nan_policy='omit'), # nan_policy='omit' dla odporności
                        lambda x: x.quantile(0.95) if not x.empty else np.nan,
                        lambda x: x.quantile(0.75) - x.quantile(0.25) if not x.empty else np.nan
                    ])
                    # Zmiana nazw kolumn po agregacji
                    stats.columns = [
                        "count", "min", "max", "mean", "median", "std", "var",
                        "skew", "kurtosis", "percentile_95", "iqr"
                    ]
                    # Użyj zdefiniowanych nazw do wyświetlenia
                    stats_display = stats.rename(columns={k: v[0] for k, v in STAT_DEFINITIONS.items() if k in stats.columns})

                    st.dataframe(stats_display.T.style.format("{:.2f}", na_rep="-")) # Transpozycja dla lepszego widoku

                    st.subheader(f"Wykresy dla: {selected_column_adv}")
                    fig_adv, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Wykres pudełkowy
                    sns.boxplot(
                        data=st.session_state.df_combined, x="Źródło", y=selected_column_adv, ax=ax1,
                        palette={st.session_state.df1_name: "#3498db", st.session_state.df2_name: "#e74c3c"}
                    )
                    ax1.set_title(f"Rozkład {selected_column_adv}")
                    ax1.tick_params(axis='x', rotation=10) # Lekkie obrócenie etykiet osi X

                    # Histogram
                    sns.histplot(
                        data=st.session_state.df_combined, x=selected_column_adv, hue="Źródło", kde=True, ax=ax2,
                        palette={st.session_state.df1_name: "#3498db", st.session_state.df2_name: "#e74c3c"}
                    )
                    ax2.set_title(f"Histogram {selected_column_adv}")

                    plt.tight_layout()
                    st.pyplot(fig_adv)
                    plt.close(fig_adv) # Ważne: Zamknij figurę po wyświetleniu

                except Exception as e:
                    st.error(f"Błąd podczas generowania statystyk lub wykresów: {e}")

    # --- Zakładka: Analiza korelacji ---
    with tabs[2]:
        key = "correlation"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns:
            st.warning("Brak kolumn numerycznych do analizy korelacji.")
        else:
            source_options_corr = [st.session_state.df1_name, st.session_state.df2_name, "Oba"]
            selected_source_corr = st.selectbox("Wybierz źródło danych:", source_options_corr, index=2, key="sel_corr_source") # Domyślnie "Oba"

            data_corr = st.session_state.df_combined
            if selected_source_corr != "Oba":
                data_corr = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == selected_source_corr]

            if data_corr.empty or data_corr[st.session_state.numeric_columns].isnull().all().all():
                 st.warning(f"Brak wystarczających danych numerycznych dla źródła: {selected_source_corr}")
            elif len(st.session_state.numeric_columns) < 2:
                 st.warning("Potrzebne są co najmniej dwie kolumny numeryczne do analizy korelacji.")
            else:
                st.subheader("Macierz korelacji (Spearman)")
                try:
                    corr = data_corr[st.session_state.numeric_columns].corr(method='spearman')

                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr,
                                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Siła korelacji'})
                    ax_corr.set_title(f"Macierz korelacji Spearmana dla: {selected_source_corr}", pad=20)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)

                    # Analiza silnych korelacji
                    st.subheader("Znaczące korelacje (|r| > 0.7)")
                    strong_corrs = []
                    for i in range(len(corr.columns)):
                        for j in range(i + 1, len(corr.columns)):
                            val = corr.iloc[i, j]
                            if abs(val) > 0.7:
                                strong_corrs.append((corr.columns[i], corr.columns[j], val))

                    if strong_corrs:
                        analysis = "Wykryto następujące silne korelacje:\n\n"
                        for var1, var2, val in strong_corrs:
                            direction = "dodatnią" if val > 0 else "ujemną"
                            strength = "BARDZO SILNĄ" if abs(val) > 0.8 else "silną"
                            analysis += (f"- **{var1} i {var2}:** korelacja {direction} ({strength}, r = {val:.2f})\n")
                            analysis += f"  - *Znaczenie:* Wzrost `{var1}` wiąże się ze {'wzrostem' if val > 0 else 'spadkiem'} `{var2}`\n\n"
                        st.markdown(analysis)
                    else:
                        st.info("Nie wykryto silnych korelacji (|r| > 0.7) między zmiennymi.")

                except Exception as e:
                    st.error(f"Błąd podczas obliczania lub wyświetlania korelacji: {e}")

    # --- Zakładka: Analiza regresji ---
    with tabs[3]:
        key = "regression"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns or len(st.session_state.numeric_columns) < 2:
            st.warning("Potrzebne są co najmniej dwie kolumny numeryczne do analizy regresji.")
        else:
            col1_regr, col2_regr = st.columns(2)
            with col1_regr:
                source_options_regr = [st.session_state.df1_name, st.session_state.df2_name, "Oba"]
                selected_source_regr = st.selectbox("Wybierz źródło danych:", source_options_regr, index=2, key="sel_regr_source")
                dep_var = st.selectbox("Wybierz zmienną zależną (Y):", st.session_state.numeric_columns, key="sel_regr_dep")

            possible_indep_vars = [col for col in st.session_state.numeric_columns if col != dep_var]

            with col2_regr:
                 if not possible_indep_vars:
                      st.warning("Brak dostępnych zmiennych niezależnych (po wyborze zmiennej zależnej).")
                      indep_vars = []
                 else:
                      indep_vars = st.multiselect("Wybierz zmienne niezależne (X):", possible_indep_vars, key="sel_regr_indep")


            if dep_var and indep_vars:
                data_regr = st.session_state.df_combined
                if selected_source_regr != "Oba":
                    data_regr = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == selected_source_regr]

                # Usuń wiersze z brakami danych w wybranych kolumnach
                cols_for_regr = [dep_var] + indep_vars
                data_regr_clean = data_regr[cols_for_regr].dropna()


                if len(data_regr_clean) < len(indep_vars) + 2: # Potrzeba więcej obserwacji niż zmiennych + stała
                     st.warning(f"Niewystarczająca liczba obserwacji ({len(data_regr_clean)}) bez braków danych dla wybranych zmiennych w źródle '{selected_source_regr}'.")
                else:
                    st.subheader(f"Wyniki regresji dla: {selected_source_regr}")
                    try:
                        Y = data_regr_clean[dep_var]
                        X = data_regr_clean[indep_vars]
                        X = sm.add_constant(X) # Dodaj stałą (intercept)

                        model = sm.OLS(Y, X).fit()

                        st.subheader("Podsumowanie modelu")
                        st.text_area("Statystyki OLS", model.summary().as_text(), height=400, key="regr_summary")

                        st.subheader("Wykresy diagnostyczne")
                        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 10))

                        # Reszty vs Dopasowane
                        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True,
                                    line_kws={'color': 'red', 'lw': 1}, ax=axes[0, 0])
                        axes[0, 0].set_title('Reszty vs Dopasowane wartości')
                        axes[0, 0].set_xlabel('Dopasowane wartości')
                        axes[0, 0].set_ylabel('Reszty')

                        # Q-Q Plot
                        sm.qqplot(model.resid, line='s', ax=axes[0, 1])
                        axes[0, 1].set_title('Normalny Q-Q plot reszt')

                        # Histogram reszt
                        sns.histplot(model.resid, kde=True, ax=axes[1, 0])
                        axes[1, 0].set_title('Rozkład reszt')
                        axes[1, 0].set_xlabel('Reszty')

                        # Obserwowane vs Dopasowane
                        axes[1, 1].scatter(model.fittedvalues, Y, alpha=0.5) # Użyj Y (rzeczywiste wartości)
                        # Linia y=x
                        min_val = min(model.fittedvalues.min(), Y.min())
                        max_val = max(model.fittedvalues.max(), Y.max())
                        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--') # Linia odniesienia y=x
                        axes[1, 1].set_title('Obserwowane vs Dopasowane wartości')
                        axes[1, 1].set_xlabel('Dopasowane wartości')
                        axes[1, 1].set_ylabel(f"Obserwowane ({dep_var})")


                        plt.tight_layout()
                        st.pyplot(fig_diag)
                        plt.close(fig_diag)

                    except Exception as e:
                        st.error(f"Błąd podczas przeprowadzania analizy regresji: {e}")
            else:
                st.info("Wybierz zmienną zależną (Y) i co najmniej jedną zmienną niezależną (X), aby przeprowadzić analizę.")

    # --- Zakładka: Diagnostyka ---
    with tabs[4]:
        key = "diagnostics"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns:
            st.warning("Brak kolumn numerycznych do przeprowadzenia diagnostyki.")
        else:
            st.subheader("Wyniki testów diagnostycznych dla poszczególnych zmiennych")

            results_diag = {}
            error_occurred = False
            for col in st.session_state.numeric_columns:
                try:
                    group1 = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == st.session_state.df1_name][col].dropna()
                    group2 = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == st.session_state.df2_name][col].dropna()
                    combined_group = st.session_state.df_combined[col].dropna()

                    # Sprawdzenie, czy mamy wystarczająco danych w każdej grupie
                    if len(group1) < 3 or len(group2) < 3 or len(combined_group) < 3:
                         results_diag[col] = {"error": f"Niewystarczająca liczba danych (wymagane min. 3 w każdej grupie i łącznie) dla kolumny '{col}'."}
                         continue # Przejdź do następnej kolumny

                    # Test Shapiro-Wilka (normalność dla połączonych danych)
                    # Wymaga co najmniej 3 próbek
                    if len(combined_group) >= 3:
                        stat_shapiro, p_shapiro = shapiro(combined_group)
                    else:
                        p_shapiro = np.nan # Nie można wykonać testu

                    # Test Levene'a (równość wariancji między grupami)
                    # Wymaga danych w obu grupach
                    if len(group1) > 0 and len(group2) > 0:
                         stat_levene, p_levene = levene(group1, group2)
                    else:
                         p_levene = np.nan

                    # ANOVA (różnica średnich między grupami)
                    if len(group1) > 0 and len(group2) > 0:
                         stat_anova, p_anova = f_oneway(group1, group2)
                    else:
                        p_anova = np.nan

                    # Test t-Studenta (Welcha - nie zakłada równości wariancji)
                    if len(group1) > 1 and len(group2) > 1: # Welch t-test wymaga min 2 obserwacji w grupie
                         stat_ttest, p_ttest = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
                    else:
                         p_ttest = np.nan

                    results_diag[col] = {
                        "Shapiro-Wilk (Normalność)": p_shapiro,
                        "Levene (Równość wariancji)": p_levene,
                        "ANOVA (Różnica średnich)": p_anova,
                        "t-test Welcha (Różnica średnich)": p_ttest
                    }

                except Exception as e:
                    results_diag[col] = {"error": f"Błąd analizy dla '{col}': {e}"}
                    error_occurred = True

            # Wyświetlanie wyników
            for col, tests in results_diag.items():
                 st.markdown(f"--- \n #### Analiza dla: `{col}`")
                 if "error" in tests:
                      st.error(tests["error"])
                 else:
                      for test_name, p_val in tests.items():
                           if np.isnan(p_val):
                                interpretation = "Nie można wykonać (za mało danych)"
                                color = "grey"
                                p_val_str = "N/A"
                           else:
                                interpretation = "**Istotne (p < 0.05)**" if p_val < 0.05 else "Nieistotne (p >= 0.05)"
                                color = "red" if p_val < 0.05 else "green"
                                p_val_str = f"{p_val:.4f}"

                           st.markdown(f"- **{test_name}:** p-value = {p_val_str} (<span style='color:{color};'>{interpretation}</span>)", unsafe_allow_html=True)
            if error_occurred:
                 st.warning("Wystąpiły błędy podczas przeprowadzania niektórych testów. Sprawdź komunikaty powyżej.")

    # --- Zakładka: Niestandardowy wykres ---
    with tabs[5]:
        key = "custom_plot"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns or len(st.session_state.numeric_columns) < 1:
            st.warning("Potrzebna jest co najmniej jedna kolumna numeryczna do stworzenia wykresu.")
        else:
            col1_cust, col2_cust, col3_cust = st.columns(3)

            with col1_cust:
                x_var = st.selectbox("Wybierz zmienną dla osi X:", st.session_state.numeric_columns, key="sel_cust_x")
                y_var = st.selectbox("Wybierz zmienną dla osi Y:", st.session_state.numeric_columns, key="sel_cust_y", index = min(1, len(st.session_state.numeric_columns)-1) ) # Domyślnie druga kolumna, jeśli istnieje

            with col2_cust:
                 source_options_cust = [st.session_state.df1_name, st.session_state.df2_name, "Oba"]
                 selected_source_cust = st.selectbox("Wybierz źródło danych:", source_options_cust, index=2, key="sel_cust_source")
                 plot_style = st.selectbox("Wybierz typ wykresu:", ["Punkty", "Punkty + Linia", "Linia"], key="sel_cust_style")


            # Sekcja linii referencyjnych
            with col3_cust:
                st.markdown("**Linie wzorcowe:**")
                ref_x = st.number_input("Wartość X dla linii pionowej:", value=None, format="%f", key="inp_cust_refx")
                ref_y = st.number_input("Wartość Y dla linii poziomej:", value=None, format="%f", key="inp_cust_refy")

                add_line_button = st.button("Dodaj linię wzorcową", key="btn_cust_addline")
                clear_lines_button = st.button("Wyczyść linie wzorcowe", key="btn_cust_clearline")

                if add_line_button:
                    if ref_x is not None or ref_y is not None:
                        st.session_state.reference_lines.append((ref_x, ref_y))
                        # Nie resetuj inputów, użytkownik może chcieć dodać więcej
                        st.rerun() # Przeładuj, aby odświeżyć wykres
                    else:
                        st.warning("Podaj wartość dla X lub Y, aby dodać linię.")

                if clear_lines_button:
                    st.session_state.reference_lines = []
                    st.rerun() # Przeładuj, aby odświeżyć wykres

                # Wyświetlanie dodanych linii
                if st.session_state.reference_lines:
                     st.caption("Dodane linie:")
                     for i, (lx, ly) in enumerate(st.session_state.reference_lines):
                          line_desc = []
                          if lx is not None: line_desc.append(f"X={lx}")
                          if ly is not None: line_desc.append(f"Y={ly}")
                          st.caption(f"- Linia {i+1}: {', '.join(line_desc)}")


            # Generowanie wykresu
            if x_var and y_var:
                data_cust = st.session_state.df_combined
                if selected_source_cust != "Oba":
                    data_cust = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == selected_source_cust]

                if data_cust.empty or data_cust[[x_var, y_var]].isnull().all().all():
                    st.warning(f"Brak wystarczających danych dla wykresu w źródle: {selected_source_cust}")
                else:
                    st.subheader(f"Wykres: {y_var} vs {x_var}")
                    try:
                        fig_cust, ax_cust = plt.subplots(figsize=(10, 6))

                        plot_kwargs = {
                            "data": data_cust,
                            "x": x_var,
                            "y": y_var,
                            "hue": "Źródło",
                            "palette": {st.session_state.df1_name: "#3498db", st.session_state.df2_name: "#e74c3c"},
                            "ax": ax_cust
                        }

                        if plot_style == "Punkty":
                             sns.scatterplot(**plot_kwargs, alpha=0.7)
                        elif plot_style == "Punkty + Linia":
                             sns.scatterplot(**plot_kwargs, alpha=0.7)
                             # Rysuj linię bez legendy, bo legenda jest już od punktów
                             sns.lineplot(**{**plot_kwargs, "legend": False, "alpha": 0.8})
                        elif plot_style == "Linia":
                             # Sortowanie danych wg osi X przed rysowaniem linii jest kluczowe
                             data_cust_sorted = data_cust.sort_values(by=x_var)
                             plot_kwargs["data"] = data_cust_sorted
                             sns.lineplot(**plot_kwargs)

                        # Dodawanie linii wzorcowych
                        line_labels_added = set() # Aby uniknąć powtarzania etykiet w legendzie
                        for i, (line_x, line_y) in enumerate(st.session_state.reference_lines):
                             if line_x is not None:
                                 label = f'Wzorzec X={line_x}' if f'Wzorzec X={line_x}' not in line_labels_added else ""
                                 ax_cust.axvline(x=line_x, color='g', linestyle='--', linewidth=1.5, label=label)
                                 if label: line_labels_added.add(label)
                             if line_y is not None:
                                  label = f'Wzorzec Y={line_y}' if f'Wzorzec Y={line_y}' not in line_labels_added else ""
                                  ax_cust.axhline(y=line_y, color='g', linestyle='--', linewidth=1.5, label=label)
                                  if label: line_labels_added.add(label)


                        ax_cust.set_title(f"Zależność między {y_var} a {x_var} ({selected_source_cust})")
                        ax_cust.set_xlabel(x_var)
                        ax_cust.set_ylabel(y_var)
                        ax_cust.grid(True, linestyle='--', alpha=0.6)
                        # Poprawienie legendy - usuń duplikaty jeśli się pojawią
                        handles, labels = ax_cust.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles)) # Słownik usuwa duplikaty kluczy (etykiet)
                        ax_cust.legend(by_label.values(), by_label.keys(), title="Źródło / Wzorzec")


                        plt.tight_layout()
                        st.pyplot(fig_cust)
                        plt.close(fig_cust)

                    except Exception as e:
                        st.error(f"Błąd podczas generowania niestandardowego wykresu: {e}")
            else:
                st.info("Wybierz zmienne dla osi X i Y.")

    # --- Zakładka: Moc vs Obroty ---
    with tabs[6]:
        key = "power_rpm"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if 'Moc_KM' not in st.session_state.df_combined.columns or 'Moc_obr' not in st.session_state.df_combined.columns:
            st.warning("Brak wymaganych kolumn 'Moc_KM' i 'Moc_obr' w załadowanych danych do analizy Moc vs Obroty.")
        else:
            col1_pr, col2_pr = st.columns(2)
            with col1_pr:
                 source_options_pr = [st.session_state.df1_name, st.session_state.df2_name, "Oba"]
                 selected_source_pr = st.selectbox("Wybierz źródło danych:", source_options_pr, index=2, key="sel_pr_source")
            with col2_pr:
                 reference_type_pr = st.selectbox("Wybierz wzorzec odniesienia:", ["1.9", "2.0"], key="sel_pr_ref")


            # Filtrowanie danych
            data_pr = st.session_state.df_combined
            if selected_source_pr != "Oba":
                data_pr = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == selected_source_pr]

            # Sprawdzenie czy dane nie są puste i czy kolumny istnieją po filtrowaniu
            if data_pr.empty or 'Moc_KM' not in data_pr.columns or 'Moc_obr' not in data_pr.columns:
                 st.warning(f"Brak danych 'Moc_KM' lub 'Moc_obr' dla źródła: {selected_source_pr}")
            else:
                 # Konwersja na numeryczne, z obsługą błędów
                 try:
                      data_pr['Moc_obr'] = pd.to_numeric(data_pr['Moc_obr'], errors='coerce')
                      data_pr['Moc_KM'] = pd.to_numeric(data_pr['Moc_KM'], errors='coerce')
                      data_pr = data_pr.dropna(subset=['Moc_obr', 'Moc_KM']) # Usuń wiersze z NaN w kluczowych kolumnach
                 except Exception as e:
                      st.error(f"Błąd konwersji danych na numeryczne w zakładce Moc vs Obroty: {e}")
                      data_pr = pd.DataFrame() # Wyczyść dane jeśli konwersja się nie udała


                 if data_pr.empty:
                      st.warning(f"Brak prawidłowych danych numerycznych 'Moc_KM' i 'Moc_obr' dla źródła: {selected_source_pr} po usunięciu błędów.")
                 else:
                    st.subheader("Wykres Mocy od Obrotów")
                    try:
                        fig_pr, ax_pr = plt.subplots(figsize=(10, 6))

                        # Sortowanie danych wg obrotów przed rysowaniem linii
                        data_pr_sorted = data_pr.sort_values(by='Moc_obr')

                        # Rysowanie danych
                        if selected_source_pr == "Oba":
                             sns.lineplot(data=data_pr_sorted, x="Moc_obr", y="Moc_KM", hue="Źródło", ax=ax_pr,
                                          palette={st.session_state.df1_name: "#3498db", st.session_state.df2_name: "#e74c3c"})
                        else:
                             sns.lineplot(data=data_pr_sorted, x="Moc_obr", y="Moc_KM", ax=ax_pr, label=selected_source_pr,
                                           color="#3498db" if selected_source_pr == st.session_state.df1_name else "#e74c3c") # Użyj odpowiedniego koloru


                        # Rysowanie krzywej wzorcowej
                        if reference_type_pr == "1.9":
                             obr_ref, moc_ref = get_reference_curve_19()
                             ref_label = "Wzorzec 1.9"
                             ref_color = "blue"
                        else:
                             obr_ref, moc_ref = get_reference_curve_20()
                             ref_label = "Wzorzec 2.0"
                             ref_color = "orange"
                        ax_pr.plot(obr_ref, moc_ref, linestyle="--", color=ref_color, label=ref_label)

                        # Dodawanie cieniowania sektorów
                        ax_pr.axvspan(3000, 4000, color="green", alpha=0.1, label="3000–4000 obr/min (Zielony)")
                        ax_pr.axvspan(4000, 5000, color="yellow", alpha=0.1, label="4000–5000 obr/min (Żółty)")
                        ax_pr.axvspan(5000, 6000, color="red", alpha=0.1, label="5000–6000 obr/min (Czerwony)")


                        ax_pr.set_title(f"Zależność mocy od obrotów ({selected_source_pr}) vs Wzorzec {reference_type_pr}")
                        ax_pr.set_xlabel("Obroty [rpm]")
                        ax_pr.set_ylabel("Moc [KM]")
                        ax_pr.grid(True, linestyle='--', alpha=0.6)

                        # Poprawienie legendy - usuń duplikaty jeśli się pojawią
                        handles, labels = ax_pr.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax_pr.legend(by_label.values(), by_label.keys(), title="Źródło / Wzorzec / Sektor")

                        plt.tight_layout()
                        st.pyplot(fig_pr)
                        plt.close(fig_pr)

                        # Analiza bezpieczeństwa (tylko dla pojedynczego silnika)
                        if selected_source_pr != "Oba":
                             st.subheader(f"Analiza bezpieczeństwa sektorów dla: {selected_source_pr}")
                             safety_results = analyze_engine_safety(selected_source_pr, data_pr_sorted, reference_type_pr) # Użyj posortowanych danych bez NaN

                             if safety_results:
                                 # Sprawdź, czy są jakieś prawidłowe wyniki
                                 valid_safety_results = [r for r in safety_results if not np.isnan(r["Wskaźnik bezpieczeństwa"])]
                                 if valid_safety_results:
                                     best_sector = valid_safety_results[0]["Sektor"] # Pierwszy po posortowaniu wg wskaźnika
                                     st.success(f"✅ Najbezpieczniejszy sektor (najniższy wskaźnik): **{best_sector}**")
                                 else:
                                      st.info("Nie udało się obliczyć wskaźnika bezpieczeństwa dla żadnego sektora (prawdopodobnie brak danych lub błędy).")


                                 # Konwertuj na DataFrame dla ładniejszego wyświetlenia
                                 safety_df = pd.DataFrame(safety_results)
                                 st.dataframe(safety_df.style.format("{:.2f}", na_rep="-").highlight_min(subset=['Wskaźnik bezpieczeństwa'], color='lightgreen', axis=0))
                             else:
                                 st.warning("Nie udało się przeprowadzić analizy bezpieczeństwa.")
                        else:
                             st.info("Analiza bezpieczeństwa jest dostępna tylko przy wyborze pojedynczego silnika.")


                    except Exception as e:
                        st.error(f"Błąd podczas generowania wykresu mocy vs obroty: {e}")


else:
    st.info("👈 Proszę wgrać pliki Excel z danymi dla obu silników w panelu bocznym i kliknąć 'Przetwórz pliki'.")