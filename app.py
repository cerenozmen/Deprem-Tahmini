import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# =========================
# Sayfa Ayarları
# =========================
st.set_page_config(
    page_title="İstanbul Deprem Tahmin Modeli",
    page_icon="🌍",
    layout="wide"
)

# =========================
# MODELLER
# =========================
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load("rf_reg_deprem_buyukluk.joblib")
        clf_model = joblib.load("rf_clf_deprem_olasilik.joblib")
        return reg_model, clf_model
    except FileNotFoundError as e:
        st.error(
            "Model dosyaları bulunamadı! "
            "Lütfen .joblib dosyalarının 'app.py' ile aynı klasörde olduğundan emin olun.\n"
            f"Hata: {e}"
        )
        return None, None

rf_reg, rf_clf = load_models()

# =========================
# İLÇE -> KOORDİNAT
# =========================
DISTRICTS = {
    "Adalar": (40.8739, 29.1236),
    "Arnavutköy": (41.1831, 28.7406),
    "Ataşehir": (40.9923, 29.1244),
    "Avcılar": (40.9792, 28.7214),
    "Bağcılar": (41.0390, 28.8564),
    "Bahçelievler": (40.9967, 28.8500),
    "Bakırköy": (40.9833, 28.8725),
    "Başakşehir": (41.0937, 28.8020),
    "Bayrampaşa": (41.0466, 28.9023),
    "Beşiktaş": (41.0422, 29.0094),
    "Beykoz": (41.1340, 29.0950),
    "Beylikdüzü": (41.0015, 28.6417),
    "Beyoğlu": (41.0369, 28.9847),
    "Büyükçekmece": (41.0205, 28.5850),
    "Çatalca": (41.1429, 28.4610),
    "Çekmeköy": (41.0352, 29.1757),
    "Esenler": (41.0437, 28.8762),
    "Esenyurt": (41.0343, 28.6801),
    "Eyüpsultan": (41.0480, 28.9330),
    "Fatih": (41.0186, 28.9390),
    "Gaziosmanpaşa": (41.0584, 28.9153),
    "Güngören": (41.0179, 28.8790),
    "Kadıköy": (40.9917, 29.0275),
    "Kağıthane": (41.0853, 28.9780),
    "Kartal": (40.9006, 29.1894),
    "Küçükçekmece": (41.0009, 28.7906),
    "Maltepe": (40.9357, 29.1551),
    "Pendik": (40.8775, 29.2356),
    "Sancaktepe": (41.0090, 29.2130),
    "Sarıyer": (41.1667, 29.0500),
    "Silivri": (41.0744, 28.2464),
    "Sultanbeyli": (40.9680, 29.2690),
    "Sultangazi": (41.1065, 28.8687),
    "Şile": (41.1755, 29.6130),
    "Şişli": (41.0602, 28.9877),
    "Tuzla": (40.8183, 29.3006),
    "Ümraniye": (41.0164, 29.1240),
    "Üsküdar": (41.0220, 29.0320),
    "Zeytinburnu": (40.9941, 28.9033),
}

district_df = pd.DataFrame(
    [{"ilce_adi": k, "lat": v[0], "lon": v[1]} for k, v in DISTRICTS.items()]
).sort_values("ilce_adi").reset_index(drop=True)

# =========================
# Deprem Kataloğu (KOD İÇİNDE)
# =========================
# time formatı: "YYYY-MM-DD HH:MM:SS"
# Buraya kendi kayıtlarını ekleyebilirsin.
QUAKES = [
    {"time": "2025-12-10 03:12:10", "lat": 40.9792, "lon": 28.7214, "mag": 3.1, "depth_km": 9.8},
    {"time": "2025-12-12 14:05:00", "lat": 40.9833, "lon": 28.8725, "mag": 2.7, "depth_km": 11.2},
    {"time": "2025-12-18 09:41:22", "lat": 40.9917, "lon": 29.0275, "mag": 3.4, "depth_km": 7.5},
    {"time": "2025-12-22 21:18:45", "lat": 41.0009, "lon": 28.7906, "mag": 3.0, "depth_km": 10.0},
    {"time": "2025-12-28 06:55:10", "lat": 41.0186, "lon": 28.9390, "mag": 2.9, "depth_km": 8.0},
    {"time": "2026-01-02 00:11:02", "lat": 41.0390, "lon": 28.8564, "mag": 3.2, "depth_km": 12.0},
]

@st.cache_data
def load_quake_catalog_from_code(quakes: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(quakes).copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon", "mag"]).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["mag"] = df["mag"].astype(float)
    if "depth_km" in df.columns:
        df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    else:
        df["depth_km"] = np.nan
    return df

quake_catalog = load_quake_catalog_from_code(QUAKES)

# =========================
# Yardımcılar
# =========================
def derive_date_features(d: datetime.date):
    return {"month": d.month, "dow": d.weekday(), "dayofyear": d.timetuple().tm_yday}

def week_dates(start_date: datetime.date, days: int = 7):
    return [start_date + datetime.timedelta(days=i) for i in range(days)]

def mag_to_energy(m: float) -> float:
    # Gutenberg–Richter yaklaşık enerji ilişkisi (erg): log10(E) = 1.5M + 4.8
    return float(10 ** (1.5 * m + 4.8))

def compute_roll30_features(quake_df: pd.DataFrame, lat_bin: float, lon_bin: float, as_of_date: datetime.date):
    start_dt = datetime.datetime.combine(as_of_date - datetime.timedelta(days=30), datetime.time.min)
    end_dt = datetime.datetime.combine(as_of_date, datetime.time.min)

    lat0, lat1 = lat_bin, lat_bin + 0.1
    lon0, lon1 = lon_bin, lon_bin + 0.1

    sub = quake_df[
        (quake_df["time"] >= start_dt) & (quake_df["time"] < end_dt) &
        (quake_df["lat"] >= lat0) & (quake_df["lat"] < lat1) &
        (quake_df["lon"] >= lon0) & (quake_df["lon"] < lon1)
    ].copy()

    cnt = float(len(sub))
    if cnt == 0:
        return {
            "roll30_count": 0.0,
            "roll30_maxmag": 0.0,
            "roll30_meanmag": 0.0,
            "roll30_depth": 0.0,
            "roll30_energy_30d": 0.0,
            "roll30_energy_rate_30d": 0.0,
        }

    maxmag = float(sub["mag"].max())
    meanmag = float(sub["mag"].mean())

    if sub["depth_km"].notna().any():
        meandepth = float(sub["depth_km"].dropna().mean())
    else:
        meandepth = 0.0

    energies = sub["mag"].apply(mag_to_energy).astype(float)
    e30 = float(energies.sum())
    er30 = float(e30 / 30.0)

    return {
        "roll30_count": cnt,
        "roll30_maxmag": maxmag,
        "roll30_meanmag": meanmag,
        "roll30_depth": meandepth,
        "roll30_energy_30d": e30,
        "roll30_energy_rate_30d": er30,
    }

# =========================
# Arayüz
# =========================
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri ile tahmin ve risk analizi yapar.")

tab1, tab2 = st.tabs(["📉 1 Haftalık Büyüklük Tahmini (Regresyon)", "⚠️ 1 Haftalık Bölgesel Risk (Sınıflandırma)"])

# =========================================================
# TAB 1: REGRESYON
# =========================================================
with tab1:
    st.header("İlçe Bazlı 1 Haftalık Büyüklük Tahmini")
    if rf_reg is None:
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        selected_district = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist())
        row = district_df[district_df["ilce_adi"] == selected_district].iloc[0]
        input_lat = float(row["lat"])
        input_lon = float(row["lon"])
        st.caption(f"Seçilen ilçe merkez koordinatı: **{input_lat:.5f}, {input_lon:.5f}**")

        # Derinlik UI'dan kaldırıldı (arkada sabit)
        default_depth_km = 10.0

        start_date = st.date_input("Başlangıç Tarihi", datetime.date.today())

    with c2:
        st.subheader("⚙️ Arka Plan Varsayılanları")
        with st.expander("Varsayılan değerleri gör / değiştir"):
            default_fault_dist = st.number_input("fault_distance (km) [default]", value=5.0)
            default_b_value = st.number_input("b_value [default]", value=1.0)
            default_log_energy = st.number_input("log_energy [default]", value=9.0)
            default_e30 = st.number_input("energy_30d [default]", value=10000.0)
            default_er30 = st.number_input("energy_rate_30d [default]", value=100.0)
            default_e90 = st.number_input("energy_90d [default]", value=50000.0)
            default_er90 = st.number_input("energy_rate_90d [default]", value=100.0)

    if st.button("1 Haftalık Tahmin Üret", type="primary"):
        dates = week_dates(start_date, 7)
        rows = []

        log_e30 = float(np.log1p(default_e30))
        log_e90 = float(np.log1p(default_e90))
        log_er30 = float(np.log1p(default_er30))
        log_er90 = float(np.log1p(default_er90))

        for d in dates:
            df_date = derive_date_features(d)
            rows.append({
                "date": d,
                "lat": input_lat,
                "lon": input_lon,
                "depth_km": float(default_depth_km),
                "fault_distance": float(default_fault_dist),
                "b_value": float(default_b_value),
                "log_energy": float(default_log_energy),
                "energy_30d": float(default_e30),
                "energy_rate_30d": float(default_er30),
                "energy_90d": float(default_e90),
                "energy_rate_90d": float(default_er90),
                "log_energy_30d": log_e30,
                "log_energy_90d": log_e90,
                "log_energy_rate_30d": log_er30,
                "log_energy_rate_90d": log_er90,
                "month": df_date["month"],
                "dow": df_date["dow"],
                "dayofyear": df_date["dayofyear"],
            })

        pred_df = pd.DataFrame(rows)
        model_input = pred_df.drop(columns=["date"]).copy()

        try:
            pred_df["pred_mw"] = rf_reg.predict(model_input)
            st.success("1 haftalık tahmin üretildi ✅")
            st.dataframe(
                pred_df[["date", "pred_mw"]].assign(pred_mw=lambda x: x["pred_mw"].round(2)),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")

# =========================================================
# TAB 2: SINIFLANDIRMA
# =========================================================
with tab2:
    st.header("İlçe Bazlı 1 Haftalık Deprem Olasılığı (M ≥ 3.0)")
    if rf_clf is None:
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        selected_district_c = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist(), key="district_clf")
        rowc = district_df[district_df["ilce_adi"] == selected_district_c].iloc[0]
        c_lat = float(rowc["lat"])
        c_lon = float(rowc["lon"])

        lat_bin = float(np.floor(c_lat / 0.1) * 0.1)
        lon_bin = float(np.floor(c_lon / 0.1) * 0.1)
        st.caption(f"Hesaplanan Hücre: **{lat_bin:.1f}, {lon_bin:.1f}**")

        start_date_c = st.date_input("Başlangıç Tarihi", datetime.date.today(), key="start_date_c")

    # Otomatik (kod içi katalogdan) hesaplanmış varsayılanlar
    auto_feats = compute_roll30_features(quake_catalog, lat_bin, lon_bin, start_date_c)

    defv = {
        "roll30_count": 5.0,
        "roll30_maxmag": 3.5,
        "roll30_meanmag": 2.5,
        "roll30_depth": 10.0,
        "roll30_energy_30d": 1000.0,
        "roll30_energy_rate_30d": 10.0
    }
    defv.update(auto_feats)

    with c2:
        # UI'da varsayılan olarak otomatik hesaplanan değerler görünür.
        # Kullanıcı isterse override eder.
        roll30_count = st.number_input("Son 30 gündeki deprem sayısı", value=float(defv["roll30_count"]))
        roll30_maxmag = st.number_input("Son 30 gündeki maks. büyüklük", value=float(defv["roll30_maxmag"]))
        roll30_meanmag = st.number_input("Son 30 gündeki ort. büyüklük", value=float(defv["roll30_meanmag"]))
        roll30_depth = st.number_input("Son 30 gündeki ort. derinlik", value=float(defv["roll30_depth"]))

        # enerji UI yok -> arkada otomatik
        roll30_energy = float(defv["roll30_energy_30d"])
        roll30_energy_rate = float(defv["roll30_energy_rate_30d"])

        st.caption("30 günlük değerler kod içindeki katalogdan otomatik dolduruldu (istersen elle değiştirebilirsin).")

    if st.button("1 Haftalık Risk Hesapla", type="primary"):
        dates = week_dates(start_date_c, 7)
        rows = []

        for d in dates:
            df_date = derive_date_features(d)

            # Her gün için rolling 30 gün otomatik hesap
            feats_d = compute_roll30_features(quake_catalog, lat_bin, lon_bin, d)

            rows.append({
                "date": d,
                "lat_bin": float(lat_bin),
                "lon_bin": float(lon_bin),
                "roll30_count": float(feats_d["roll30_count"]),
                "roll30_maxmag": float(feats_d["roll30_maxmag"]),
                "roll30_meanmag": float(feats_d["roll30_meanmag"]),
                "roll30_depth": float(feats_d["roll30_depth"]),
                "roll30_energy_30d": float(feats_d["roll30_energy_30d"]),
                "roll30_energy_rate_30d": float(feats_d["roll30_energy_rate_30d"]),
                "month": df_date["month"],
                "dow": df_date["dow"],
                "dayofyear": df_date["dayofyear"],
            })

        pred_df = pd.DataFrame(rows)
        model_input = pred_df.drop(columns=["date"]).copy()

        try:
            pred_df["prob"] = rf_clf.predict_proba(model_input)[:, 1]
            show_df = pred_df[["date", "prob"]].copy()
            show_df["prob_%"] = (show_df["prob"] * 100).round(2)
            st.success("1 haftalık risk üretildi ✅")
            st.dataframe(show_df.drop(columns=["prob"]), use_container_width=True)
        except Exception as e:
            st.error(f"Sınıflandırma hatası: {e}")

st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
