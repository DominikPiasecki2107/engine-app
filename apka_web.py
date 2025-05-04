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
from datetime import datetime # Do tworzenia przykładowych dat

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Kompleksowa analiza silników")
st.title("Kompleksowa analiza silników - Wersja Webowa")

# --- Definicje (takie same jak w oryginale + nowa zakładka) ---
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
    "user_manual": { # NOWA ZAKŁADKA
        "title": "Instrukcja obsługi",
        "description": "Szczegółowy przewodnik po funkcjonalnościach aplikacji oraz metodologii analitycznej."
    },
    # --- NOWA DEFINICJA ZAKŁADKI ---
    "example_analysis": {
        "title": "Przykładowa analiza",
        "description": "Przewodnik krok po kroku pokazujący, jak interpretować wyniki analizy na podstawie przykładowych danych."
    },
    # --- KONIEC NOWEJ DEFINICJI ---
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

# --- Przykładowe Dane ---
def generate_sample_data():
    data1 = {
        'Opis': ['TEST1', 'TEST1', 'TEST1', 'TEST1', 'TEST1'],
        'Moc_KM': [48.1, 75.9, 113.1, 126.3, 145.6],
        'Moc_obr': [3000, 4000, 4750, 5500, 6000],
        'Mom_NM': [150.5, 166.8, 193.0, 193.4, 205.7],
        'Mom_obr': [3020, 3000, 3900, 3490, 4700],
        'Temperatura(°C)': [23.6, 24.2, 19.2, 22.6, 23.4],
        'Ciśnienie(hPa)': [983.4, 991.7, 993.9, 995.6, 994.9],
        'Norma_DIN': [1.035, 1.046, 1.030, 1.035, 1.030],
        'W2': [1950, 1950, 1950, 1950, 1950],
        'W1': [1900, 1900, 1900, 1900, 1900],
        'Data': [datetime(2025, 1, 2, 10, 0), datetime(2025, 1, 2, 10, 5), datetime(2025, 1, 2, 10, 10), datetime(2025, 1, 2, 10, 15), datetime(2025, 1, 2, 10, 20)],
        'Wsp_prze': [30.36, 30.36, 25.08, 25.08, 25.08],
        'Bieg': [5, 5, 5, 5, 5]
    }
    df1 = pd.DataFrame(data1)

    data2 = {
        'Opis': ['TEST2', 'TEST2', 'TEST2', 'TEST2', 'TEST2', 'TEST2'],
        'Moc_KM': [65.7, 70.4, 75.9, 88.8, 115.6, 149.9],
        'Moc_obr': [3250, 3500, 3750, 4500, 5000, 5750],
        'Mom_NM': [155.0, 168.8, 152.2, 190.0, 197.1, 201.5],
        'Mom_obr': [3020, 3080, 3030, 3930, 3950, 4400],
        'Temperatura(°C)': [23.4, 24.8, 24.3, 19.7, 22.3, 22.8],
        'Ciśnienie(hPa)': [985.8, 989.9, 989.9, 991.9, 999.2, 995.2],
        'Norma_DIN': [1.035, 1.047, 1.046, 1.030, 1.036, 1.036],
        'W2': [1950, 1950, 1950, 1950, 1950, 1950],
        'W1': [1900, 1900, 1900, 1900, 1900, 1900],
        'Data': [datetime(2025, 1, 3, 11, 0), datetime(2025, 1, 3, 11, 5), datetime(2025, 1, 3, 11, 10), datetime(2025, 1, 3, 11, 15), datetime(2025, 1, 3, 11, 20), datetime(2025, 1, 3, 11, 25)],
        'Wsp_prze': [30.36, 30.36, 30.36, 25.08, 25.08, 25.08],
        'Bieg': [5, 5, 5, 5, 5, 5]
    }
    df2 = pd.DataFrame(data2)
    return df1, df2

# Generowanie przykładowych danych
sample_df1, sample_df2 = generate_sample_data()
sample_df1_name = "Silnik Wzorcowy"
sample_df2_name = "Silnik Testowy"

# Połączenie przykładowych danych do niektórych analiz opisowych
sample_df1_copy = sample_df1.copy()
sample_df2_copy = sample_df2.copy()
sample_df1_copy["Źródło"] = sample_df1_name
sample_df2_copy["Źródło"] = sample_df2_name
sample_df_combined = pd.concat([sample_df1_copy, sample_df2_copy], ignore_index=True)
sample_numeric_cols = sample_df_combined.select_dtypes(include=np.number).columns.tolist()

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

    # Dla zakładki 'Instrukcja obsługi' i 'Przykładowa analiza' nie pokazujemy dodatkowych informacji w expanderze
    if tab_key in ["user_manual", "example_analysis"]:
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
                 # Konwersja z obsługą różnych formatów i błędów
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

def analyze_engine_safety(engine_name, df_engine_data, reference_type):
    """Analizuje bezpieczeństwo silnika porównując do krzywej wzorcowej."""
    # Ta funkcja oczekuje DataFrame zawierającego TYLKO dane dla jednego silnika
    # i z usuniętymi NaN w kolumnach mocy/obrotów.
    if df_engine_data is None or df_engine_data.empty:
         st.warning(f"Brak danych dla silnika: {engine_name} przekazanych do analyze_engine_safety.")
         return []
    if 'Moc_KM' not in df_engine_data.columns or 'Moc_obr' not in df_engine_data.columns:
         st.warning("Brak wymaganych kolumn ('Moc_KM', 'Moc_obr') w danych przekazanych do analyze_engine_safety.")
         return []

    # Sprawdzenie typów danych i próba konwersji - powinno już być zrobione, ale dla pewności
    try:
        obroty = df_engine_data["Moc_obr"].astype(float).values
        moc = df_engine_data["Moc_KM"].astype(float).values
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
        min_obr_real = obr_real.min()
        max_obr_real = obr_real.max()
        min_obr_ref = obroty_ref.min()
        max_obr_ref = obroty_ref.max()

        if min_obr_real < min_obr_ref or max_obr_real > max_obr_ref:
             st.warning(f"Zakres obrotów w danych ({min_obr_real}-{max_obr_real}) wykracza poza zakres krzywej referencyjnej ({min_obr_ref}-{max_obr_ref}) dla sektora {start}-{end}. Wyniki mogą być niemiarodajne.")
             # Interpolacja z obsługą wartości poza zakresem (ustawi na NaN)
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
            # Znormalizujmy trochę składowe, aby miały podobny rząd wielkości (opcjonalne, ale może pomóc)
            # Przykładowa normalizacja: podziel przez średnią moc w sektorze (jeśli nie jest zerem)
            # norm_sr_roznica = sr_roznica / srednia if srednia != 0 else sr_roznica
            # norm_std = std / srednia if srednia != 0 else std
            # norm_rozstep = rozstep / srednia if srednia != 0 else rozstep
            # ocena = 0.4 * norm_sr_roznica + 0.4 * norm_std + 0.2 * norm_rozstep
            # LUB prościej bez normalizacji:
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

    Wersja: 1.3 (Web) - Dodano Przykładową analizę
    Autor: [Dominik Piasecki] / Konwersja i modyfikacja: AI Gemini

    Funkcjonalności:
    - Wczytywanie danych z dwóch plików Excel.
    - Podgląd surowych danych.
    - Szczegółowa analiza statystyczna.
    - Analiza korelacji (heatmapa Spearmana).
    - Analiza regresji liniowej (OLS) z diagnostyką.
    - Testy diagnostyczne (normalność, równość wariancji, ANOVA, t-test).
    - Tworzenie niestandardowych wykresów z liniami wzorcowymi.
    - Analiza zależności mocy od obrotów z krzywą wzorcową i oceną bezpieczeństwa.
    - Szczegółowa instrukcja obsługi.
    - **Nowość:** Przykładowa analiza z interpretacją.

    © 2025 Wszelkie prawa zastrzeżone (dla oryginalnej aplikacji).
    """)

if clear_button:
    clear_data()
    st.rerun() # Wymuś przeładowanie strony, aby wyczyścić stan

# Logika przetwarzania danych po naciśnięciu przycisku
if process_button:
    if uploaded_file1 and uploaded_file2 and name1 and name2:
        try:
            # Użyj io.BytesIO do odczytu danych z pamięci
            bytes_data1 = io.BytesIO(uploaded_file1.getvalue())
            bytes_data2 = io.BytesIO(uploaded_file2.getvalue())
            df1 = pd.read_excel(bytes_data1)
            df2 = pd.read_excel(bytes_data2)

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
            st.sidebar.caption("Upewnij się, że pliki są w formacie .xlsx i zawierają prawidłowe dane.")
    else:
        st.sidebar.warning("Proszę wgrać oba pliki i podać obie nazwy.")

# --- Główna część aplikacji (Zakładki) ---
# Sprawdź, czy dane zostały wgrane przez użytkownika LUB czy jest to pierwszy raz (jeszcze nic nie wgrano)
data_loaded = st.session_state.df_combined is not None and not st.session_state.df_combined.empty

# --- NOWA LOGIKA: Dodaj klucz 'example_analysis' ---
tab_keys = ["user_manual", "example_analysis", "raw_data", "advanced", "correlation", "regression", "diagnostics", "custom_plot", "power_rpm"]

# Filtruj klucze zakładek - pokaż wszystkie oprócz analitycznych, jeśli dane nie są załadowane
if not data_loaded:
    # Pokaż tylko instrukcję i przykład, jeśli nie ma danych
    active_tab_keys = ["user_manual", "example_analysis"]
else:
    # Pokaż wszystkie zakładki, jeśli dane są załadowane
    active_tab_keys = tab_keys

# Generuj tytuły tylko dla aktywnych zakładek
tab_titles = [TAB_DEFINITIONS.get(k, {}).get("title", k.replace("_", " ").title()) for k in active_tab_keys]

# Utwórz zakładki
tabs = st.tabs(tab_titles)

# --- Zakładka: Instrukcja obsługi ---
user_manual_tab_index = active_tab_keys.index("user_manual") if "user_manual" in active_tab_keys else -1
if user_manual_tab_index != -1:
    with tabs[user_manual_tab_index]:
        key = "user_manual"
        st.header(TAB_DEFINITIONS[key]['title'])
        st.markdown(TAB_DEFINITIONS[key]['description'])
        st.markdown("---")

        st.subheader("1. Wprowadzenie")
        st.markdown("""
        Aplikacja **Kompleksowa analiza silników - Wersja Webowa** stanowi interaktywne narzędzie do zaawansowanej analizy porównawczej danych pochodzących z dwóch różnych zbiorów, reprezentujących np. dwa silniki lub dwa różne stany tego samego silnika. Umożliwia ona eksplorację danych, identyfikację wzorców, testowanie hipotez statystycznych oraz wizualizację zależności między zmiennymi.
        """)

        st.subheader("2. Przygotowanie i Wczytanie Danych")
        st.markdown("""
        Podstawowym krokiem jest dostarczenie danych wejściowych. Aplikacja akceptuje dwa pliki w formacie Microsoft Excel (`.xlsx`). Każdy plik powinien zawierać dane dotyczące jednego z analizowanych obiektów (np. Silnik 1 i Silnik 2).

        **Wymagania dotyczące danych:**
        - **Format:** Pliki `.xlsx`.
        - **Struktura:** Dane powinny być zorganizowane w formie tabelarycznej, gdzie kolumny reprezentują mierzone parametry (zmienne), a wiersze - poszczególne obserwacje lub pomiary. Zaleca się stosowanie opisowych nazw kolumn.
        - **Kolumny numeryczne:** Do przeprowadzenia większości analiz (statystyki opisowe, korelacja, regresja, testy diagnostyczne) niezbędne są kolumny zawierające dane numeryczne. Aplikacja automatycznie identyfikuje takie kolumny.
        - **Kolumny specjalne (opcjonalne, ale zalecane dla pełnej funkcjonalności):**
            - `Moc_KM`: Kolumna zawierająca wartości mocy w koniach mechanicznych.
            - `Moc_obr`: Kolumna zawierająca wartości obrotów na minutę, odpowiadające pomiarom mocy.
            - `Data`: Kolumna z datą lub znacznikiem czasu (aplikacja spróbuje przekonwertować ją na format daty).

        **Proces wczytywania:**
        1. W panelu bocznym (po lewej stronie) zlokalizuj sekcję **"Wczytywanie danych"**.
        2. Użyj przycisku **"Wybierz plik Excel dla Silnika 1"**, aby załadować pierwszy plik.
        3. W polu tekstowym **"Nazwa dla Silnika 1"** wprowadź identyfikator dla pierwszego zbioru danych (domyślnie "Silnik 1"). Nazwa ta będzie używana w legendach wykresów i tabelach.
        4. Powtórz kroki 2 i 3 dla drugiego zbioru danych (**"Wybierz plik Excel dla Silnika 2"** i **"Nazwa dla Silnika 2"**).
        5. Kliknij przycisk **"Przetwórz pliki"**. Aplikacja wczyta dane, połączy je w jeden zbiorczy DataFrame, doda kolumnę `Źródło` identyfikującą pochodzenie danych i przygotuje interfejs do dalszej analizy.
        6. W przypadku sukcesu pojawi się komunikat w panelu bocznym. W razie błędów (np. nieprawidłowy format pliku) zostanie wyświetlone stosowne ostrzeżenie.
        7. Aby rozpocząć nową analizę z innymi plikami, użyj przycisku **"Wyczyść dane"**. Spowoduje to usunięcie aktualnie załadowanych danych i zresetowanie stanu aplikacji.
        """)

        st.subheader("3. Opis Funkcjonalności (Zakładki)")
        st.markdown("Po poprawnym wczytaniu danych, dostępne stają się następujące moduły analityczne w formie zakładek:")

        with st.expander("3.1 Surowe dane"):
            st.markdown("""
            Umożliwia szybki przegląd początkowych wierszy (domyślnie 10) każdego z załadowanych zbiorów danych. Służy do weryfikacji poprawności wczytania i struktury plików.
            - **Wyświetlane informacje:** Fragment tabeli danych dla każdego z silników.
            """)

        with st.expander("3.2 Analiza statystyczna"):
            st.markdown("""
            Prezentuje podstawowe i zaawansowane statystyki opisowe dla wybranej zmiennej numerycznej, porównując rozkłady między dwoma źródłami danych.
            - **Interakcja:** Wybierz interesującą Cię kolumnę z listy rozwijanej.
            - **Wyświetlane informacje:**
                - **Tabela statystyk:** Zawiera miary tendencji centralnej (średnia, mediana), miary rozproszenia (odchylenie standardowe, wariancja, rozstęp międzykwartylowy, min, max), miary kształtu rozkładu (skośność, kurtoza) oraz percentyl 95. Definicje poszczególnych miar dostępne są w sekcji informacyjnej zakładki (ikona ℹ️).
                - **Wykres pudełkowy (Box Plot):** Wizualizuje rozkład danych dla obu grup, pokazując medianę, kwartyle (Q1, Q3), zakres danych (z wyłączeniem wartości odstających) oraz potencjalne wartości odstające (outliery). Pozwala na szybką ocenę różnic w położeniu i rozproszeniu danych między grupami.
                - **Histogram z krzywą gęstości (KDE):** Przedstawia częstość występowania wartości w określonych przedziałach dla obu grup. Nałożona krzywa gęstości (Kernel Density Estimate) wygładza histogram, sugerując kształt rozkładu prawdopodobieństwa zmiennej.
            """)

        with st.expander("3.3 Analiza korelacji"):
            st.markdown("""
            Bada siłę i kierunek liniowej zależności między parami zmiennych numerycznych przy użyciu współczynnika korelacji Spearmana. Metoda ta jest odporna na wartości odstające i nie wymaga założenia o normalności rozkładu danych.
            - **Interakcja:** Możliwość wyboru analizy dla jednego silnika lub obu łącznie.
            - **Wyświetlane informacje:**
                - **Mapa cieplna (Heatmap):** Graficzna reprezentacja macierzy korelacji. Kolory komórek wskazują na siłę i kierunek korelacji (zgodnie z legendą kolorów, zazwyczaj od niebieskiego/zimnego dla korelacji ujemnych do czerwonego/ciepłego dla dodatnich). Wartości liczbowe współczynników są naniesione na komórki.
                - **Lista silnych korelacji:** Automatycznie identyfikuje i wypisuje pary zmiennych, dla których bezwzględna wartość współczynnika korelacji Spearmana przekracza próg 0.7, interpretując je jako silne zależności.
            """)

        with st.expander("3.4 Analiza regresji"):
            st.markdown("""
            Umożliwia budowę i ocenę modelu regresji liniowej metodą najmniejszych kwadratów (Ordinary Least Squares - OLS). Celem jest wyjaśnienie zmienności jednej zmiennej (zależnej Y) za pomocą jednej lub więcej zmiennych (niezależnych X).
            - **Interakcja:**
                - Wybór źródła danych (jeden silnik lub oba).
                - Wybór zmiennej zależnej (Y) z listy kolumn numerycznych.
                - Wybór jednej lub więcej zmiennych niezależnych (X) spośród pozostałych kolumn numerycznych.
            - **Wyświetlane informacje:**
                - **Podsumowanie modelu OLS:** Szczegółowe wyniki estymacji modelu, w tym: współczynniki regresji (coefficients) wraz z ich błędami standardowymi, statystykami t i wartościami p (istotność statystyczna), współczynnik determinacji $R^2$ (miara dopasowania modelu), skorygowany $R^2$, statystyka F (test istotności całego modelu) oraz inne miary diagnostyczne (np. AIC, BIC).
                - **Wykresy diagnostyczne:**
                    - *Reszty vs Dopasowane:* Pozwala ocenić założenie o homoskedastyczności (stałej wariancji reszt) oraz liniowości modelu. Idealnie, punkty powinny być losowo rozproszone wokół linii poziomej y=0.
                    - *Normalny Q-Q plot reszt:* Służy do oceny założenia o normalności rozkładu reszt. Punkty powinny układać się wzdłuż linii prostej.
                    - *Histogram reszt:* Wizualna reprezentacja rozkładu błędów modelu. Powinien przypominać rozkład normalny (kształt dzwonu).
                    - *Obserwowane vs Dopasowane:* Porównuje rzeczywiste wartości zmiennej Y z wartościami przewidzianymi przez model. Punkty powinny grupować się wokół linii y=x.
            """)

        with st.expander("3.5 Diagnostyka"):
            st.markdown("""
            Zawiera zestaw testów statystycznych służących do weryfikacji podstawowych założeń dotyczących danych, co jest kluczowe dla poprawnej interpretacji wyników innych analiz (np. ANOVA, t-test). Testy przeprowadzane są dla każdej zmiennej numerycznej.
            - **Wyświetlane informacje (dla każdej zmiennej):**
                - **Test Shapiro-Wilka (Normalność):** Testuje hipotezę zerową, że dane pochodzą z populacji o rozkładzie normalnym. Istotny wynik (p < 0.05) sugeruje odrzucenie tej hipotezy. Przeprowadzany na połączonych danych.
                - **Test Levene'a (Równość wariancji):** Testuje hipotezę zerową, że wariancje w obu grupach (silnikach) są równe (homoskedastyczność). Istotny wynik (p < 0.05) wskazuje na istotne różnice w wariancjach (heteroskedastyczność).
                - **ANOVA (Różnica średnich):** Jednoczynnikowa analiza wariancji testuje hipotezę zerową, że średnie wartości zmiennej w obu grupach są równe. Istotny wynik (p < 0.05) sugeruje, że co najmniej jedna średnia różni się od drugiej.
                - **Test t-Studenta (Welcha):** Testuje hipotezę zerową o równości średnich w dwóch grupach, nie zakładając równości wariancji (dlatego stosowana jest poprawka Welcha). Jest bardziej szczegółowy niż ANOVA dla dwóch grup. Istotny wynik (p < 0.05) wskazuje na statystycznie istotną różnicę między średnimi.
            - **Interpretacja:** Wyniki są prezentowane wraz z wartością p i kolorowym oznaczeniem istotności statystycznej (czerwony dla p < 0.05, zielony dla p >= 0.05). W przypadku niewystarczającej ilości danych do przeprowadzenia testu, wyświetlany jest odpowiedni komunikat.
            """)

        with st.expander("3.6 Niestandardowy wykres"):
            st.markdown("""
            Moduł do tworzenia własnych wizualizacji zależności między dwiema wybranymi zmiennymi numerycznymi.
            - **Interakcja:**
                - Wybór zmiennej dla osi X.
                - Wybór zmiennej dla osi Y.
                - Wybór źródła danych (jeden silnik, drugi, lub oba).
                - Wybór typu wykresu: punkty (scatter plot), punkty z linią (łączącą średnie lub wg kolejności, zależnie od danych), tylko linia.
                - **Linie wzorcowe:** Możliwość dodania poziomych (Y=const) lub pionowych (X=const) linii referencyjnych do wykresu poprzez wpisanie wartości i kliknięcie "Dodaj linię wzorcową". Linie można usuwać przyciskiem "Wyczyść linie wzorcowe".
            - **Wyświetlane informacje:** Wykres zależności Y od X zgodnie z wybranymi parametrami, z możliwością rozróżnienia grup kolorami i dodanymi liniami referencyjnymi. Legenda wyjaśnia znaczenie kolorów i linii.
            """)

        with st.expander("3.7 Moc vs Obroty"):
            st.markdown("""
            Specjalistyczna zakładka dedykowana analizie charakterystyki mocy silnika w funkcji prędkości obrotowej. Wymaga obecności kolumn `Moc_KM` i `Moc_obr` w danych wejściowych.
            - **Interakcja:**
                - Wybór źródła danych (jeden silnik lub oba).
                - Wybór krzywej wzorcowej do porównania (np. dla silnika 1.9 lub 2.0 - krzywe są predefiniowane w kodzie).
            - **Wyświetlane informacje:**
                - **Wykres Mocy od Obrotów:** Prezentuje zależność mocy (KM) od obrotów (rpm) dla wybranych danych. Na wykresie naniesiona jest również wybrana krzywa wzorcowa oraz cieniowane obszary reprezentujące trzy sektory obrotów (3000-4000, 4000-5000, 5000-6000 rpm) używane w analizie bezpieczeństwa.
                - **Analiza bezpieczeństwa sektorów (tylko dla pojedynczego silnika):** Tabela przedstawiająca wyniki analizy porównawczej rzeczywistej charakterystyki mocy z krzywą wzorcową w zdefiniowanych sektorach obrotów. Zawiera:
                    - Średnią moc, odchylenie standardowe mocy i rozstęp mocy w danym sektorze.
                    - Średnią bezwzględną różnicę między mocą rzeczywistą a mocą wzorcową (interpolowaną dla tych samych obrotów).
                    - **Wskaźnik bezpieczeństwa:** Syntetyczna miara obliczona jako ważona suma średniej różnicy do wzorca, odchylenia standardowego i rozstępu. Niższa wartość wskaźnika sugeruje większą stabilność i zgodność z wzorcem w danym sektorze. Sektor z najniższym wskaźnikiem jest wyróżniony.
            """)

        st.subheader("4. Metodologia i Założenia")
        st.markdown("""
        Aplikacja wykorzystuje standardowe biblioteki języka Python do analizy danych i wizualizacji (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`). Przyjęto następujące główne założenia metodologiczne:

        - **Statystyki opisowe:** Obliczane są standardowe miary tendencji centralnej, rozproszenia i kształtu rozkładu.
        - **Korelacja:** Wykorzystywany jest nieparametryczny współczynnik korelacji rang Spearmana, który mierzy siłę i kierunek monotonicznej zależności między zmiennymi.
        - **Regresja liniowa:** Stosowana jest metoda najmniejszych kwadratów (OLS). Interpretacja wyników wymaga spełnienia założeń OLS, takich jak liniowość zależności, normalność rozkładu reszt, homoskedastyczność reszt oraz brak współliniowości zmiennych niezależnych. Wykresy diagnostyczne pomagają w ocenie tych założeń.
        - **Testy diagnostyczne:**
            - Test Shapiro-Wilka jest stosowany do oceny normalności.
            - Test Levene'a służy do oceny homoskedastyczności (równości wariancji między grupami).
            - ANOVA i test t-Studenta (Welcha) służą do porównywania średnich między grupami. Wybór testu t Welcha podyktowany jest jego odpornością na nierówność wariancji.
        - **Analiza bezpieczeństwa (Moc vs Obroty):** Rzeczywiste dane mocy są porównywane z predefiniowaną krzywą wzorcową poprzez interpolację wartości wzorcowych dla obrotów obecnych w danych rzeczywistych. Wskaźnik bezpieczeństwa jest heurystyczną miarą łączącą odchylenie od wzorca ze stabilnością (odchylenie standardowe) i zakresem zmienności (rozstęp) mocy w danym sektorze.

        **Ograniczenia:**
        - Jakość analizy jest silnie zależna od jakości i kompletności danych wejściowych.
        - Interpretacja wyników testów statystycznych i modeli regresji wymaga wiedzy dziedzinowej i statystycznej. Aplikacja dostarcza narzędzi, ale ostateczna interpretacja należy do użytkownika.
        - Predefiniowane krzywe wzorcowe mają charakter przykładowy i mogą wymagać dostosowania do specyfiki analizowanych silników.
        """)

        st.subheader("5. Informacje o Wersji")
        st.markdown("""
        - **Wersja:** 1.3 (Web)
        - **Główne biblioteki:** Streamlit, Pandas, Matplotlib, Seaborn, Scipy, Statsmodels
        - **Kontakt / Autor oryginalny:** Dominik Piasecki (wg sekcji "O programie")
        - **Modyfikacje (wersja web + instrukcja + przykład):** AI Gemini
        """)

# --- NOWA ZAKŁADKA: Przykładowa analiza ---
example_analysis_tab_index = active_tab_keys.index("example_analysis") if "example_analysis" in active_tab_keys else -1
if example_analysis_tab_index != -1:
    with tabs[example_analysis_tab_index]:
        key = "example_analysis"
        st.header(TAB_DEFINITIONS[key]['title'])
        st.markdown(TAB_DEFINITIONS[key]['description'])
        st.markdown("---")

        st.info("""
        **Cel tej zakładki:** Pokazanie, jak można interpretować wyniki generowane przez aplikację.
        Używamy tutaj **wbudowanych, przykładowych danych** dla dwóch hipotetycznych silników:
        * `Silnik Wzorcowy`
        * `Silnik Testowy`

        **Pamiętaj:** Analizę na swoich danych przeprowadzasz w pozostałych zakładkach po ich wgraniu!
        """)

        st.subheader("1. Przykładowe Dane")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{sample_df1_name} (fragment):**")
            st.dataframe(sample_df1.head(3))
        with col2:
            st.markdown(f"**{sample_df2_name} (fragment):**")
            st.dataframe(sample_df2.head(3))

        st.markdown("---")
        st.subheader("2. Interpretacja Wyników (Przykłady)")

        # --- Przykład: Analiza Statystyczna ---
        with st.expander("Przykład: Interpretacja - Analiza Statystyczna (z zakładki 'Analiza statystyczna')"):
            st.markdown(f"""
            Załóżmy, że w zakładce **'Analiza statystyczna'** wybraliśmy kolumnę `Moc_KM`. Zobaczymy tam:

            * **Tabela Statystyk:** Porównujemy wartości dla `{sample_df1_name}` i `{sample_df2_name}`.
                * `Średnia (mean)`: Np. `{sample_df1_name}` ma średnią moc 102.8 KM, a `{sample_df2_name}` 93.1 KM. Widzimy, że silnik wzorcowy osiąga średnio wyższą moc w tych pomiarach.
                * `Odch. standardowe (std)`: Np. `{sample_df1_name}` ma std 38.5 KM, a `{sample_df2_name}` 30.2 KM. Moc silnika wzorcowego jest bardziej zróżnicowana (większy rozrzut) w tych danych.
                * `Mediana (median)`: Porównanie mediany (wartości środkowej) może dać inny obraz niż średnia, jeśli występują wartości skrajne. Np. mediana 113.1 KM dla wzorcowego vs 82.3 KM dla testowego.
                * `Min` / `Max`: Pokazują zakresy mocy dla obu silników.
                * `Skośność (skew)` i `Kurtoza (kurtosis)`: Informują o kształcie rozkładu mocy (czy jest symetryczny, czy spiczasty).

            * **Wykres Pudełkowy:** Wizualnie porównuje rozkłady.
                * Położenie "pudełek" (IQR) pokazuje, gdzie koncentruje się środkowe 50% danych. Jeśli pudełko `{sample_df1_name}` jest wyżej niż `{sample_df2_name}`, sugeruje to generalnie wyższe wartości mocy dla tego silnika.
                * Długość "wąsów" i ewentualne punkty poza nimi (outliery) pokazują rozrzut i wartości skrajne. Szersze pudełko lub dłuższe wąsy oznaczają większą zmienność.

            * **Histogram z KDE:** Pokazuje, które wartości mocy występują najczęściej.
                * Położenie "szczytów" histogramów wskazuje dominujące zakresy mocy.
                * Kształt krzywej KDE (gęstości) pokazuje ogólny rozkład - czy jest jednomodalny (jeden szczyt), bimodalny (dwa szczyty), itp.

            **Wniosek (przykład):** *Na podstawie tych (przykładowych!) danych, Silnik Wzorcowy wydaje się osiągać wyższą moc maksymalną i średnią, ale jego wyniki są bardziej rozrzucone niż Silnika Testowego.*
            """)

        # --- Przykład: Analiza Korelacji ---
        with st.expander("Przykład: Interpretacja - Analiza Korelacji (z zakładki 'Analiza korelacji')"):
            st.markdown(f"""
            W zakładce **'Analiza korelacji'**, po wybraniu np. 'Oba' źródła, zobaczymy mapę cieplną.

            * **Mapa Cieplna (Heatmap):** Patrzymy na kolory i wartości w komórkach.
                * Interesuje nas np. korelacja między `Moc_KM` a `Mom_NM`. Jeśli komórka na ich przecięciu jest czerwona (blisko +1), oznacza to **silną korelację dodatnią**. Gdy moc rośnie, moment obrotowy też zwykle rośnie. W naszych przykładowych danych korelacja Spearmana wynosi ok. 0.99 - bardzo silna dodatnia.
                * Korelacja `Moc_KM` i `Temperatura(°C)`: Jeśli komórka jest bliska 0 (biała/jasna), oznacza to **brak lub bardzo słabą korelację liniową**. Temperatura nie wydaje się być silnie powiązana z mocą w tych danych. W przykładzie korelacja to ok. 0.12 - bardzo słaba.
                * Korelacja `Moc_obr` i `Ciśnienie(hPa)`: Jeśli komórka jest niebieska (blisko -1), oznacza to **korelację ujemną**. Gdy jedna wartość rośnie, druga maleje. W przykładzie korelacja to ok. 0.14 - brak istotnej korelacji.

            * **Lista Silnych Korelacji:** Aplikacja wypisze pary zmiennych, gdzie wartość bezwzględna korelacji przekracza 0.7. W naszym przykładzie na pewno pojawiłaby się para `Moc_KM` i `Mom_NM`.

            **Wniosek (przykład):** *W tych danych obserwujemy bardzo silny związek między mocą a momentem obrotowym (co jest oczekiwane). Temperatura czy ciśnienie nie wykazują silnego liniowego związku z mocą czy obrotami.*
            """)

        # --- Przykład: Diagnostyka ---
        with st.expander("Przykład: Interpretacja - Diagnostyka (z zakładki 'Diagnostyka')"):
            st.markdown(f"""
            Zakładka **'Diagnostyka'** dostarcza wyników testów statystycznych dla każdej zmiennej numerycznej, porównując `{sample_df1_name}` i `{sample_df2_name}`. Kluczowa jest **wartość p (p-value)**.

            * Przyjmuje się próg istotności **p = 0.05**.
                * Jeśli **p < 0.05 (kolor czerwony, istotne)**: Odrzucamy hipotezę zerową (H0). Oznacza to, że **jest** statystycznie istotna różnica lub zależność (zależnie od testu).
                * Jeśli **p >= 0.05 (kolor zielony, nieistotne)**: Nie ma podstaw do odrzucenia H0. Oznacza to, że **nie ma** dowodów na statystycznie istotną różnicę lub zależność.

            * **Przykład dla zmiennej `Temperatura(°C)`:**
                * `Shapiro-Wilk (Normalność)`: Testuje, czy *połączone* dane temperatury z obu silników pochodzą z rozkładu normalnego. Jeśli p < 0.05, rozkład nie jest normalny.
                * `Levene (Równość wariancji)`: Testuje, czy wariancje (rozrzut) temperatury są **podobne** w obu grupach. Jeśli p < 0.05, wariancje **różnią się** istotnie.
                * `ANOVA` / `t-test Welcha (Różnica średnich)`: Testują, czy średnia temperatura **różni się** istotnie między silnikami. Jeśli p < 0.05, średnie **są różne**.

            * **Przykład dla zmiennej `Moc_KM`:**
                 * `Levene`: Załóżmy, że p = 0.04. Oznacza to, że wariancja (rozrzut) mocy **różni się** istotnie między silnikami (co zauważyliśmy też w statystykach opisowych).
                 * `t-test Welcha`: Załóżmy, że p = 0.03. Oznacza to, że średnia moc **różni się** istotnie statystycznie między `{sample_df1_name}` a `{sample_df2_name}`.

            **Wniosek (przykład):** *Testy diagnostyczne mogą potwierdzić obserwacje z innych zakładek. Np. istotny wynik testu t dla Mocy_KM potwierdza, że różnica w średniej mocy między silnikami nie jest przypadkowa (na poziomie istotności 0.05). Istotny test Levene'a sugeruje, że porównując średnie, powinniśmy używać testu odpornego na różne wariancje (jak test t Welcha, który jest tu domyślny).*
            """)

        # --- Przykład: Moc vs Obroty ---
        with st.expander("Przykład: Interpretacja - Moc vs Obroty (z zakładki 'Moc vs Obroty')"):
            st.markdown(f"""
            Zakładka **'Moc vs Obroty'** wymaga kolumn `Moc_KM` i `Moc_obr`.

            * **Wykres Mocy od Obrotów:**
                * Pokazuje przebieg krzywej mocy dla wybranego silnika (lub obu) na tle obrotów.
                * Linia przerywana to wybrana **krzywa wzorcowa** (np. dla silnika 1.9 lub 2.0). Możemy wizualnie ocenić, jak bardzo rzeczywisty przebieg odbiega od wzorca.
                * Kolorowe tła oznaczają **sektory obrotów** (3-4k, 4-5k, 5-6k rpm) używane w analizie bezpieczeństwa.

            * **Analiza Bezpieczeństwa Sektorów (dla pojedynczego silnika):**
                * Tabela pokazuje wyniki dla każdego sektora. Porównujemy np. `{sample_df1_name}` z wzorcem `1.9`.
                * `Śr. moc`, `Odch. std`, `Rozstęp`: Statystyki mocy silnika w danym sektorze. Niskie `Odch. std` i `Rozstęp` oznaczają stabilną moc w tym zakresie obrotów.
                * `Śr. różnica do wzorca`: Jak bardzo średnio moc w sektorze odbiega od krzywej wzorcowej. Im niższa wartość, tym bliżej wzorca.
                * `Wskaźnik bezpieczeństwa`: Łączy różnicę do wzorca, odchylenie i rozstęp. **Niższa wartość oznacza większą stabilność i zgodność z wzorcem w danym sektorze.** Aplikacja podświetla wiersz z najniższym wskaźnikiem.
                * Załóżmy, że dla `{sample_df1_name}` najniższy wskaźnik bezpieczeństwa (np. 15.2) wystąpił w sektorze `4000-5000 obr/min`.

            **Wniosek (przykład):** *Wizualnie krzywa mocy dla Silnika Wzorcowego może np. przebiegać blisko wzorca 1.9 w środkowym zakresie obrotów, a odbiegać bardziej przy wyższych. Analiza bezpieczeństwa potwierdza to, wskazując sektor 4000-5000 obr/min jako najbardziej stabilny i najbliższy wzorcowi dla tego silnika w tych (przykładowych!) danych.*
            """)

        st.markdown("---")
        st.success("To tylko kilka przykładów interpretacji. Zachęcamy do eksploracji wszystkich zakładek i analizy własnych danych!")


# --- Pozostałe zakładki (korzystają z danych wgranych przez użytkownika lub pokazują informację o ich braku) ---
if data_loaded:
    # --- Zakładka: Surowe dane ---
    raw_data_tab_index = active_tab_keys.index("raw_data")
    with tabs[raw_data_tab_index]:
        key = "raw_data"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)
        st.subheader(f"Podgląd danych: {st.session_state.df1_name}")
        st.dataframe(st.session_state.df1.head(10))
        st.subheader(f"Podgląd danych: {st.session_state.df2_name}")
        st.dataframe(st.session_state.df2.head(10))

    # --- Zakładka: Analiza statystyczna ---
    advanced_tab_index = active_tab_keys.index("advanced")
    with tabs[advanced_tab_index]:
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
                    # Sprawdzenie, czy kolumna faktycznie istnieje w połączonym DataFrame
                    if selected_column_adv not in st.session_state.df_combined.columns:
                         st.error(f"Wybrana kolumna '{selected_column_adv}' nie istnieje w danych.")
                    else:
                        # Obliczanie statystyk
                        stats = st.session_state.df_combined.groupby("Źródło")[selected_column_adv].agg([
                            "count", "min", "max", "mean", "median", "std", "var",
                            "skew", lambda x: kurtosis(x, nan_policy='omit'),
                            lambda x: x.quantile(0.95) if not x.empty and pd.api.types.is_numeric_dtype(x) and x.count() > 0 else np.nan,
                            lambda x: x.quantile(0.75) - x.quantile(0.25) if not x.empty and pd.api.types.is_numeric_dtype(x) and x.count() >= 2 else np.nan # IQR wymaga min 2 punktów
                        ]).rename(columns={ # Zmiana nazw kolumn od razu
                            "<lambda_0>": "kurtosis",
                            "<lambda_1>": "percentile_95",
                            "<lambda_2>": "iqr"
                         })

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
                    st.error(f"Błąd podczas generowania statystyk lub wykresów dla '{selected_column_adv}': {e}")
                    st.error("Upewnij się, że ta kolumna zawiera dane numeryczne.")


    # --- Zakładka: Analiza korelacji ---
    correlation_tab_index = active_tab_keys.index("correlation")
    with tabs[correlation_tab_index]:
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

            # Wybierz tylko kolumny numeryczne PRZED sprawdzeniem pustych danych
            numeric_data_corr = data_corr.select_dtypes(include=np.number)

            if numeric_data_corr.empty or numeric_data_corr.isnull().all().all():
                 st.warning(f"Brak wystarczających danych numerycznych dla źródła: {selected_source_corr}")
            elif numeric_data_corr.shape[1] < 2:
                 st.warning(f"Potrzebne są co najmniej dwie kolumny numeryczne ({numeric_data_corr.shape[1]} znaleziono) do analizy korelacji.")
            else:
                st.subheader("Macierz korelacji (Spearman)")
                try:
                    corr = numeric_data_corr.corr(method='spearman')

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
                    # Iteruj tylko po dostępnych kolumnach numerycznych
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
    regression_tab_index = active_tab_keys.index("regression")
    with tabs[regression_tab_index]:
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

                # Usuń wiersze z brakami danych TYLKO w wybranych kolumnach
                cols_for_regr = [dep_var] + indep_vars
                if not all(col in data_regr.columns for col in cols_for_regr):
                     st.error(f"Jedna lub więcej wybranych kolumn ({cols_for_regr}) nie istnieje w danych źródła '{selected_source_regr}'.")
                else:
                    data_regr_clean = data_regr[cols_for_regr].dropna()

                    if len(data_regr_clean) < len(indep_vars) + 2: # Potrzeba więcej obserwacji niż zmiennych + stała
                         st.warning(f"Niewystarczająca liczba obserwacji ({len(data_regr_clean)}) bez braków danych dla wybranych zmiennych w źródle '{selected_source_regr}'. Potrzeba co najmniej {len(indep_vars) + 2}.")
                    else:
                        st.subheader(f"Wyniki regresji dla: {selected_source_regr}")
                        try:
                            Y = data_regr_clean[dep_var]
                            X = data_regr_clean[indep_vars]
                            X = sm.add_constant(X) # Dodaj stałą (intercept)

                            model = sm.OLS(Y, X).fit()

                            st.subheader("Podsumowanie modelu")
                            # Użyj st.code zamiast st.text_area dla lepszego formatowania
                            st.code(model.summary().as_text(), language='text')

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
    diagnostics_tab_index = active_tab_keys.index("diagnostics")
    with tabs[diagnostics_tab_index]:
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
                    if col not in st.session_state.df_combined.columns:
                        results_diag[col] = {"error": f"Kolumna '{col}' nie istnieje w danych."}
                        continue

                    group1 = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == st.session_state.df1_name][col].dropna()
                    group2 = st.session_state.df_combined[st.session_state.df_combined["Źródło"] == st.session_state.df2_name][col].dropna()
                    combined_group = st.session_state.df_combined[col].dropna()

                    # Inicjalizacja wyników dla kolumny
                    col_results = {}

                    # Test Shapiro-Wilka (normalność dla połączonych danych)
                    if len(combined_group) >= 3:
                        try:
                            # Sprawdź, czy dane nie są stałe (wszystkie wartości takie same)
                            if len(combined_group.unique()) > 1:
                                stat_shapiro, p_shapiro = shapiro(combined_group)
                                col_results["Shapiro-Wilk (Normalność)"] = p_shapiro
                            else:
                                col_results["Shapiro-Wilk (Normalność)"] = np.nan # Dane stałe, test nie ma sensu
                                # st.caption(f"Test Shapiro-Wilka pominięty dla '{col}' - dane są stałe.")
                        except Exception as e_shapiro:
                            col_results["Shapiro-Wilk (Normalność)"] = np.nan # Błąd testu
                            # st.caption(f"Błąd testu Shapiro-Wilka dla '{col}': {e_shapiro}") # Opcjonalny debug
                    else:
                        col_results["Shapiro-Wilk (Normalność)"] = np.nan # Za mało danych

                    # Test Levene'a (równość wariancji między grupami)
                    # Wymaga co najmniej 1 obserwacji w każdej grupie, ale scipy może wymagać więcej w praktyce
                    if len(group1) >= 2 and len(group2) >= 2:
                        try:
                             # Sprawdź, czy w grupach są różne wartości
                             if len(group1.unique()) > 1 or len(group2.unique()) > 1:
                                stat_levene, p_levene = levene(group1, group2)
                                col_results["Levene (Równość wariancji)"] = p_levene
                             else:
                                col_results["Levene (Równość wariancji)"] = np.nan # Dane stałe w jednej z grup
                                # st.caption(f"Test Levene'a pominięty dla '{col}' - dane stałe w grupie.")
                        except Exception as e_levene:
                             col_results["Levene (Równość wariancji)"] = np.nan
                             # st.caption(f"Błąd testu Levene'a dla '{col}': {e_levene}")
                    else:
                         col_results["Levene (Równość wariancji)"] = np.nan # Za mało danych

                    # ANOVA (różnica średnich między grupami)
                    # Wymaga co najmniej 1 obserwacji w każdej grupie
                    if len(group1) > 0 and len(group2) > 0:
                        try:
                            if len(group1.unique()) > 1 or len(group2.unique()) > 1:
                                 stat_anova, p_anova = f_oneway(group1, group2)
                                 col_results["ANOVA (Różnica średnich)"] = p_anova
                            else:
                                col_results["ANOVA (Różnica średnich)"] = np.nan
                                # st.caption(f"ANOVA pominięta dla '{col}' - dane stałe w grupie.")
                        except Exception as e_anova:
                             col_results["ANOVA (Różnica średnich)"] = np.nan
                             # st.caption(f"Błąd ANOVA dla '{col}': {e_anova}")
                    else:
                        col_results["ANOVA (Różnica średnich)"] = np.nan

                    # Test t-Studenta (Welcha - nie zakłada równości wariancji)
                    # Wymaga min 2 obserwacji w każdej grupie
                    if len(group1) >= 2 and len(group2) >= 2:
                        try:
                             if len(group1.unique()) > 1 or len(group2.unique()) > 1:
                                 stat_ttest, p_ttest = ttest_ind(group1, group2, equal_var=False, nan_policy='omit') # nan_policy='omit' dodane dla pewności
                                 col_results["t-test Welcha (Różnica średnich)"] = p_ttest
                             else:
                                 col_results["t-test Welcha (Różnica średnich)"] = np.nan
                                 # st.caption(f"t-test Welcha pominięty dla '{col}' - dane stałe w grupie.")
                        except Exception as e_ttest:
                             col_results["t-test Welcha (Różnica średnich)"] = np.nan
                             # st.caption(f"Błąd t-testu Welcha dla '{col}': {e_ttest}")
                    else:
                         col_results["t-test Welcha (Różnica średnich)"] = np.nan

                    # Sprawdzenie, czy wszystkie testy dały NaN (sugeruje problem z danymi dla tej kolumny)
                    if col_results and all(np.isnan(p) for p in col_results.values()):
                         results_diag[col] = {"error": f"Nie można było przeprowadzić żadnego testu dla kolumny '{col}'. Sprawdź, czy zawiera wystarczającą liczbę prawidłowych danych liczbowych w obu grupach (min. 2) i czy dane nie są stałe."}
                    elif col_results:
                         results_diag[col] = col_results
                    # else: nie dodawaj pustego wyniku, jeśli nic się nie dało zrobić

                except Exception as e:
                    results_diag[col] = {"error": f"Nieoczekiwany błąd analizy dla '{col}': {e}"}
                    error_occurred = True

            # Wyświetlanie wyników
            if not results_diag:
                st.info("Brak wyników diagnostycznych do wyświetlenia. Sprawdź, czy dane numeryczne spełniają minimalne wymagania testów.")
            else:
                for col, tests in results_diag.items():
                     st.markdown(f"--- \n #### Analiza dla: `{col}`")
                     if "error" in tests:
                          st.error(tests["error"])
                     else:
                          for test_name, p_val in tests.items():
                               if pd.isna(p_val): # Użyj pd.isna dla numpy.nan
                                   interpretation = "Nie można wykonać (brak danych / dane stałe / błąd)"
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
    custom_plot_tab_index = active_tab_keys.index("custom_plot")
    with tabs[custom_plot_tab_index]:
        key = "custom_plot"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        if not st.session_state.numeric_columns or len(st.session_state.numeric_columns) < 1:
            st.warning("Potrzebna jest co najmniej jedna kolumna numeryczna do stworzenia wykresu.")
        else:
            col1_cust, col2_cust, col3_cust = st.columns(3)

            with col1_cust:
                 # Domyślny indeks y_var zmieniony na min(1, len(...) - 1) aby uniknąć błędu, gdy jest tylko 1 kolumna
                y_index = min(1, len(st.session_state.numeric_columns) - 1) if len(st.session_state.numeric_columns) > 1 else 0 # Poprawka dla 1 kolumny
                x_var = st.selectbox("Wybierz zmienną dla osi X:", st.session_state.numeric_columns, key="sel_cust_x")
                y_var = st.selectbox("Wybierz zmienną dla osi Y:", st.session_state.numeric_columns, key="sel_cust_y", index=y_index)


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

                # Sprawdź, czy wybrane kolumny istnieją w przefiltrowanych danych i czy mają dane
                if x_var not in data_cust.columns or y_var not in data_cust.columns or data_cust[[x_var, y_var]].dropna().empty:
                     st.warning(f"Brak wystarczających danych dla '{x_var}' lub '{y_var}' w źródle: {selected_source_cust}")
                elif data_cust.empty:
                     st.warning(f"Brak danych dla źródła: {selected_source_cust}")
                else:
                    st.subheader(f"Wykres: {y_var} vs {x_var}")
                    try:
                        fig_cust, ax_cust = plt.subplots(figsize=(10, 6))

                        # Usunięto 'label' z plot_kwargs
                        plot_kwargs = {
                            "data": data_cust,
                            "x": x_var,
                            "y": y_var,
                            "hue": "Źródło" if selected_source_cust == "Oba" else None, # Hue tylko jeśli wybrano "Oba"
                            "palette": {st.session_state.df1_name: "#3498db", st.session_state.df2_name: "#e74c3c"} if selected_source_cust == "Oba" else None,
                            "ax": ax_cust
                        }

                        if plot_style == "Punkty":
                            # Dodaj label bezpośrednio tutaj, jeśli potrzebne
                            if selected_source_cust != "Oba":
                                sns.scatterplot(**plot_kwargs, label=selected_source_cust, alpha=0.7)
                            else:
                                sns.scatterplot(**plot_kwargs, alpha=0.7) # Bez label, hue się tym zajmie
                        elif plot_style == "Punkty + Linia":
                             # Dodaj label bezpośrednio tutaj, jeśli potrzebne
                            if selected_source_cust != "Oba":
                                sns.scatterplot(**plot_kwargs, label=selected_source_cust, alpha=0.7)
                            else:
                                sns.scatterplot(**plot_kwargs, alpha=0.7) # Bez label, hue się tym zajmie

                             # Rysuj linię bez legendy, bo legenda jest już od punktów, sortuj dane
                            if selected_source_cust == "Oba":
                                for name, group in data_cust.groupby("Źródło"):
                                    group_sorted = group.sort_values(by=x_var)
                                    line_color = plot_kwargs["palette"].get(name) if plot_kwargs.get("palette") else None
                                    if not group_sorted.empty: # Dodatkowe sprawdzenie
                                        sns.lineplot(data=group_sorted, x=x_var, y=y_var, ax=ax_cust, legend=False,
                                                     color=line_color, alpha=0.8, estimator=None, errorbar=None) # Dodano estimator i errorbar=None
                            else:
                                data_cust_sorted = data_cust.sort_values(by=x_var)
                                line_color = "#3498db" if selected_source_cust == st.session_state.df1_name else "#e74c3c"
                                if not data_cust_sorted.empty: # Dodatkowe sprawdzenie
                                     sns.lineplot(data=data_cust_sorted, x=x_var, y=y_var, ax=ax_cust, legend=False,
                                                  color=line_color, alpha=0.8, estimator=None, errorbar=None) # Dodano estimator i errorbar=None


                        elif plot_style == "Linia":
                             if selected_source_cust == "Oba":
                                 for name, group in data_cust.groupby("Źródło"):
                                      group_sorted = group.sort_values(by=x_var)
                                      line_color = plot_kwargs["palette"].get(name) if plot_kwargs.get("palette") else None
                                      if not group_sorted.empty: # Dodatkowe sprawdzenie
                                         sns.lineplot(data=group_sorted, x=x_var, y=y_var, ax=ax_cust, label=name,
                                                      color=line_color, estimator=None, errorbar=None) # Dodano estimator i errorbar=None
                             else:
                                  data_cust_sorted = data_cust.sort_values(by=x_var)
                                  line_color = "#3498db" if selected_source_cust == st.session_state.df1_name else "#e74c3c"
                                  if not data_cust_sorted.empty: # Dodatkowe sprawdzenie
                                       sns.lineplot(data=data_cust_sorted, x=x_var, y=y_var, ax=ax_cust, label=selected_source_cust,
                                                    color=line_color, estimator=None, errorbar=None) # Dodano estimator i errorbar=None

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

                        # Poprawienie legendy - zbierz etykiety z wykresów i linii wzorcowych
                        handles, labels = ax_cust.get_legend_handles_labels()
                        # Usuń duplikaty zachowując kolejność (ważne dla kolorów)
                        by_label = {}
                        for handle, label in zip(handles, labels):
                             if label and label not in by_label: # Sprawdź czy label nie jest pusty
                                  by_label[label] = handle
                        # Ustaw legendę tylko jeśli są jakieś etykiety
                        if by_label:
                            ax_cust.legend(by_label.values(), by_label.keys(), title="Źródło / Wzorzec")


                        plt.tight_layout()
                        st.pyplot(fig_cust)
                        plt.close(fig_cust)

                    except Exception as e:
                        st.error(f"Błąd podczas generowania niestandardowego wykresu dla '{x_var}' vs '{y_var}': {e}")
            else:
                st.info("Wybierz zmienne dla osi X i Y.")

    # --- Zakładka: Moc vs Obroty ---
    power_rpm_tab_index = active_tab_keys.index("power_rpm")
    with tabs[power_rpm_tab_index]:
        key = "power_rpm"
        st.header(TAB_DEFINITIONS[key]['title'])
        show_tab_info(key)

        # Sprawdzenie istnienia kolumn bezpośrednio w połączonym DataFrame przed filtrowaniem
        required_cols = ['Moc_KM', 'Moc_obr']
        required_cols_exist = all(col in st.session_state.df_combined.columns for col in required_cols)

        if not required_cols_exist:
            st.warning(f"Brak wymaganych kolumn ({', '.join(required_cols)}) w załadowanych danych do analizy Moc vs Obroty.")
        else:
            col1_pr, col2_pr = st.columns(2)
            with col1_pr:
                 source_options_pr = [st.session_state.df1_name, st.session_state.df2_name, "Oba"]
                 selected_source_pr = st.selectbox("Wybierz źródło danych:", source_options_pr, index=2, key="sel_pr_source")
            with col2_pr:
                 reference_type_pr = st.selectbox("Wybierz wzorzec odniesienia:", ["1.9", "2.0"], key="sel_pr_ref")


            # Filtrowanie danych
            data_pr_full = st.session_state.df_combined.copy()
            data_pr = data_pr_full # Domyślnie użyj wszystkich
            if selected_source_pr != "Oba":
                data_pr = data_pr_full[data_pr_full["Źródło"] == selected_source_pr]


            # Konwersja na numeryczne i usuwanie NaN TYLKO dla wymaganych kolumn
            try:
                 data_pr['Moc_obr'] = pd.to_numeric(data_pr['Moc_obr'], errors='coerce')
                 data_pr['Moc_KM'] = pd.to_numeric(data_pr['Moc_KM'], errors='coerce')
                 # Usuń wiersze z NaN tylko w tych kolumnach PO potencjalnym filtrowaniu wg źródła
                 data_pr = data_pr.dropna(subset=['Moc_obr', 'Moc_KM'])
            except Exception as e:
                 st.error(f"Błąd konwersji danych na numeryczne w zakładce Moc vs Obroty: {e}")
                 data_pr = pd.DataFrame() # Wyczyść dane jeśli konwersja się nie udała


            if data_pr.empty:
                 st.warning(f"Brak prawidłowych danych numerycznych 'Moc_KM' i 'Moc_obr' dla źródła: {selected_source_pr} po usunięciu błędów.")
            else:
                st.subheader("Wykres Mocy od Obrotów")
                try:
                    fig_pr, ax_pr = plt.subplots(figsize=(10, 6))

                    # Rysowanie danych
                    if selected_source_pr == "Oba":
                         # Sortuj każdą grupę osobno przed rysowaniem
                         for name, group in data_pr.groupby("Źródło"):
                              group_sorted = group.sort_values(by='Moc_obr')
                              if not group_sorted.empty: # Sprawdzenie
                                 sns.lineplot(data=group_sorted, x="Moc_obr", y="Moc_KM", ax=ax_pr, label=name,
                                              color="#3498db" if name == st.session_state.df1_name else "#e74c3c",
                                              estimator=None, errorbar=None) # Dodano estimator i errorbar=None
                    else:
                         # Sortuj pojedyncze źródło
                         data_pr_sorted = data_pr.sort_values(by='Moc_obr')
                         if not data_pr_sorted.empty: # Sprawdzenie
                             sns.lineplot(data=data_pr_sorted, x="Moc_obr", y="Moc_KM", ax=ax_pr, label=selected_source_pr,
                                          color="#3498db" if selected_source_pr == st.session_state.df1_name else "#e74c3c",
                                          estimator=None, errorbar=None) # Dodano estimator i errorbar=None


                    # Rysowanie krzywej wzorcowej
                    if reference_type_pr == "1.9":
                         obr_ref, moc_ref = get_reference_curve_19()
                         ref_label = "Wzorzec 1.9"
                         ref_color = "black"
                    else:
                         obr_ref, moc_ref = get_reference_curve_20()
                         ref_label = "Wzorzec 2.0"
                         ref_color = "grey"
                    ax_pr.plot(obr_ref, moc_ref, linestyle="--", color=ref_color, label=ref_label)

                    # Dodawanie cieniowania sektorów z etykietami
                    sector_labels_added = False # Flaga do dodania etykiet tylko raz
                    for start, end, color, sector_label in [(3000, 4000, "green", "Sektor 3-4k"),
                                                             (4000, 5000, "yellow", "Sektor 4-5k"),
                                                             (5000, 6000, "red", "Sektor 5-6k")]:
                        label_to_add = sector_label # Zmieniono - zawsze dodawaj etykietę sektora
                        ax_pr.axvspan(start, end, color=color, alpha=0.1, label=label_to_add)


                    ax_pr.set_title(f"Zależność mocy od obrotów ({selected_source_pr}) vs Wzorzec {reference_type_pr}")
                    ax_pr.set_xlabel("Obroty [rpm]")
                    ax_pr.set_ylabel("Moc [KM]")
                    ax_pr.grid(True, linestyle='--', alpha=0.6)

                    # Poprawienie legendy - usuń duplikaty i dodaj tytuł
                    handles, labels = ax_pr.get_legend_handles_labels()
                    by_label = {}
                    for handle, label in zip(handles, labels):
                        if label and label not in by_label:
                            by_label[label] = handle
                    if by_label:
                         # Umieść legendę poza obszarem wykresu
                         ax_pr.legend(by_label.values(), by_label.keys(), title="Legenda", bbox_to_anchor=(1.05, 1), loc='upper left')


                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Dostosuj tight_layout, aby zrobić miejsce na legendę
                    st.pyplot(fig_pr)
                    plt.close(fig_pr)

                    # Analiza bezpieczeństwa (tylko dla pojedynczego silnika)
                    if selected_source_pr != "Oba":
                         st.subheader(f"Analiza bezpieczeństwa sektorów dla: {selected_source_pr}")
                         # Przekazujemy tylko dane dla wybranego silnika (data_pr) do funkcji
                         safety_results = analyze_engine_safety(selected_source_pr, data_pr, reference_type_pr)

                         if safety_results:
                             valid_safety_results = [r for r in safety_results if pd.notna(r["Wskaźnik bezpieczeństwa"])]
                             if valid_safety_results:
                                 sorted_valid_results = sorted(valid_safety_results, key=lambda x: x["Wskaźnik bezpieczeństwa"])
                                 best_sector = sorted_valid_results[0]["Sektor"]
                                 st.success(f"✅ Najbardziej stabilny sektor (najniższy wskaźnik): **{best_sector}**")
                             else:
                                  st.info("Nie udało się obliczyć wskaźnika bezpieczeństwa dla żadnego sektora (prawdopodobnie brak danych lub błędy).")

                             safety_df = pd.DataFrame(safety_results)
                             numeric_safety_cols = [col for col in safety_df.columns if col != 'Sektor']

                             if valid_safety_results:
                                 st.dataframe(safety_df.style.format("{:.2f}", subset=numeric_safety_cols, na_rep="-")
                                              .highlight_min(subset=['Wskaźnik bezpieczeństwa'], color='lightgreen', axis=0, props='font-weight:bold;'))
                             else:
                                 st.dataframe(safety_df.style.format("{:.2f}", subset=numeric_safety_cols, na_rep="-"))

                         else:
                             st.warning("Nie udało się przeprowadzić analizy bezpieczeństwa (funkcja zwróciła pustą listę).")
                    else:
                         st.info("Analiza bezpieczeństwa jest dostępna tylko przy wyborze pojedynczego silnika.")


                except Exception as e:
                    st.error(f"Błąd podczas generowania wykresu mocy vs obroty lub analizy bezpieczeństwa: {e}")


# Jeśli dane nie są załadowane, wyświetl tylko komunikat
elif not data_loaded:
     # Komunikat wyświetlany jest tylko wtedy, gdy nie ma aktywnych zakładek (co nie powinno się zdarzyć, bo zawsze jest instrukcja i przykład)
     # Ale zostawiamy dla bezpieczeństwa
     if not tabs:
        st.info("👈 Proszę wgrać pliki Excel z danymi dla obu silników w panelu bocznym i kliknąć 'Przetwórz pliki'.")
        st.markdown("---")
        # Opcjonalnie wyświetl tylko instrukcję, jeśli dane nie są załadowane i nie ma innych zakładek
        st.header(TAB_DEFINITIONS['user_manual']['title'])
        st.markdown(TAB_DEFINITIONS['user_manual']['description'])
