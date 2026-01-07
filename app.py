import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math

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
# Buraya kendi verisetinden örnekleri eklemelisin (ne kadar çok, o kadar iyi)
QUAKES = [
    {"time": "2025-12-10 03:12:10", "lat": 40.98, "lon": 28.72, "mag": 3.1, "depth_km": 9.8},
    {"time": "2025-12-11 06:40:00", "lat": 41.00, "lon": 28.65, "mag": 2.8, "depth_km": 12.0},
    {"time": "2025-12-13 14:05:00", "lat": 40.99, "lon": 28.88, "mag": 2.7, "depth_km": 11.2},
    {"time": "2025-12-18 09:41:22", "lat": 41.00, "lon": 28.79, "mag": 3.4, "depth_km": 7.5},
    {"time": "2025-12-22 21:18:45", "lat": 40.99, "lon": 28.90, "mag": 3.0, "depth_km": 10.0},
    {"time": "2025-12-28 06:55:10", "lat": 41.02, "lon": 28.94, "mag": 2.9, "depth_km": 8.0},
    {"time": "2026-01-03 00:11:02", "lat": 41.04, "lon": 28.86, "mag": 3.2, "depth_km": 12.0},

    {"time": "2025-12-14 07:15:00", "lat": 40.99, "lon": 29.03, "mag": 2.8, "depth_km": 9.5},
    {"time": "2025-12-19 23:50:00", "lat": 41.02, "lon": 29.03, "mag": 3.1, "depth_km": 11.0},
    {"time": "2025-12-26 16:10:00", "lat": 40.94, "lon": 29.16, "mag": 3.3, "depth_km": 13.0},
    {"time": "2025-12-30 10:45:00", "lat": 40.90, "lon": 29.19, "mag": 3.0, "depth_km": 10.0},
    {"time": "2026-01-04 09:10:00", "lat": 40.88, "lon": 29.24, "mag": 2.7, "depth_km": 8.0},
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
# Fay hattı (KOD İÇİNDE) - örnek polylines
# =========================
# Eğer elinde gerçek fay koordinatları varsa burayı onunla değiştir.
FAULT_POINTS = [
    (40.75, 28.20),  # Silivri açıkları (yaklaşık)
    (40.75, 28.60),
    (40.78, 29.00),
    (40.80, 29.40),  # Tuzla açıkları (yaklaşık)
]

# =========================
# Yardımcılar
# =========================
def derive_date_features(d: datetime.date):
    return {"month": d.month, "dow": d.weekday(), "dayofyear": d.timetuple().tm_yday}

def week_dates(start_date: datetime.date, days: int = 7):
    return [start_date + datetime.timedelta(days=i) for i in range(days)]

def mag_to_energy(m: float) -> float:
    # Gutenberg–Richter enerji ilişkisi (erg): log10(E) = 1.5M + 4.8
    return float(10 ** (1.5 * m + 4.8))

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def fault_distance_km(lat: float, lon: float) -> float:
    # En yakın fay noktasına mesafe (basit yaklaşım)
    return float(min(haversine_km(lat, lon, fp[0], fp[1]) for fp in FAULT_POINTS))

def b_value_mle(mags: np.ndarray, mmin: float = 0.0) -> float:
    """
    Gutenberg–Richter b-value (Aki, 1965) MLE:
      b = log10(e) / (mean(M) - (Mmin - dM/2))
    Burada dM=0.1 varsayıyoruz.
    """
    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]
    mags = mags[mags >= mmin]
    if mags.size < 2:
        return 1.0
    dM = 0.1
    denom = float(np.mean(mags) - (mmin - dM / 2.0))
    if denom <= 0:
        return 1.0
    return float(np.log10(np.e) / denom)

def window_events(df: pd.DataFrame, center_lat: float, center_lon: float,
                  start_dt: datetime.datetime, end_dt: datetime.datetime,
                  radius_km: float = 30.0) -> pd.DataFrame:
    # Zaman + yarıçap filtresi
    sub = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].copy()
    if len(sub) == 0:
        return sub
    sub["_dist_km"] = sub.apply(lambda r: haversine_km(center_lat, center_lon, float(r["lat"]), float(r["lon"])), axis=1)
    return sub[sub["_dist_km"] <= radius_km].copy()

def compute_reg_features_from_dataset(quake_df: pd.DataFrame, center_lat: float, center_lon: float,
                                      as_of_date: datetime.date,
                                      radius_km: float = 30.0) -> dict:
    """
    Tab1 için ilçeye özel:
      - fault_distance
      - b_value
      - log_energy
      - energy_30d / rate
      - energy_90d / rate
    Hepsi katalogdan türetilir (CSV yok).
    """
    day0 = datetime.datetime.combine(as_of_date, datetime.time.min)

    # 30 gün
    s30 = day0 - datetime.timedelta(days=30)
    sub30 = window_events(quake_df, center_lat, center_lon, s30, day0, radius_km=radius_km)
    mags30 = sub30["mag"].to_numpy() if len(sub30) else np.array([])
    e30 = float(sub30["mag"].apply(mag_to_energy).sum()) if len(sub30) else 0.0
    er30 = float(e30 / 30.0)

    # 90 gün
    s90 = day0 - datetime.timedelta(days=90)
    sub90 = window_events(quake_df, center_lat, center_lon, s90, day0, radius_km=radius_km)
    e90 = float(sub90["mag"].apply(mag_to_energy).sum()) if len(sub90) else 0.0
    er90 = float(e90 / 90.0)

    # b_value (30 gün üzerinden)
    # mmin: burada 0 aldım, istersen eğitimde kullandığın eşik neyse onu koy
    bval = b_value_mle(mags30, mmin=0.0)

    # log_energy: burada 30 günlük toplam enerji üzerinden
    log_energy = float(np.log1p(e30))

    return {
        "fault_distance": fault_distance_km(center_lat, center_lon),
        "b_value": float(bval),
        "log_energy": float(log_energy),
        "energy_30d": float(e30),
        "energy_rate_30d": float(er30),
        "energy_90d": float(e90),
        "energy_rate_90d": float(er90),
    }

# =========================
# Arayüz
# =========================
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri ile tahmin ve risk analizi yapar.")

tab1, tab2 = st.tabs([
    "📉 1 Haftalık Büyüklük Tahmini (Regresyon)",
    "⚠️ 1 Haftalık Bölgesel Risk (Sınıflandırma)"
])

# =========================================================
# TAB 1: REGRESYON
# =========================================================
with tab1:
    st.header("İlçe Bazlı 1 Haftalık Büyüklük Tahmini")
    if rf_reg is None:
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        selected_district = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist(), key="district_reg")
        row = district_df[district_df["ilce_adi"] == selected_district].iloc[0]
        input_lat = float(row["lat"])
        input_lon = float(row["lon"])
        st.caption(f"Seçilen ilçe merkez koordinatı: **{input_lat:.5f}, {input_lon:.5f}**")

        # Derinlik UI'dan kaldırıldı (arkada sabit)
        default_depth_km = 10.0

        start_date = st.date_input("Başlangıç Tarihi", datetime.date.today(), key="start_date_reg")

    # ✅ İlçeye göre verisetinden otomatik hesap (katalogdan)
    auto_reg = compute_reg_features_from_dataset(
        quake_catalog, input_lat, input_lon, start_date,
        radius_km=30.0
    )

    # İlçe / tarih değişince state'e yaz ki UI güncellensin
    reg_ctx = f"{selected_district}|{start_date.isoformat()}"
    if st.session_state.get("last_reg_ctx") != reg_ctx:
        st.session_state["last_reg_ctx"] = reg_ctx
        st.session_state["fault_distance_key"] = float(auto_reg["fault_distance"])
        st.session_state["b_value_key"] = float(auto_reg["b_value"])
        st.session_state["log_energy_key"] = float(auto_reg["log_energy"])
        st.session_state["energy_30d_key"] = float(auto_reg["energy_30d"])
        st.session_state["energy_rate_30d_key"] = float(auto_reg["energy_rate_30d"])
        st.session_state["energy_90d_key"] = float(auto_reg["energy_90d"])
        st.session_state["energy_rate_90d_key"] = float(auto_reg["energy_rate_90d"])

    with c2:
        st.subheader("⚙️ Arka Plan Varsayılanları")
        with st.expander("Varsayılan değerleri gör / değiştir"):
            default_fault_dist = st.number_input("fault_distance (km) [auto]", key="fault_distance_key")
            default_b_value = st.number_input("b_value [auto]", key="b_value_key")
            default_log_energy = st.number_input("log_energy [auto]", key="log_energy_key")
            default_e30 = st.number_input("energy_30d [auto]", key="energy_30d_key")
            default_er30 = st.number_input("energy_rate_30d [auto]", key="energy_rate_30d_key")
            default_e90 = st.number_input("energy_90d [auto]", key="energy_90d_key")
            default_er90 = st.number_input("energy_rate_90d [auto]", key="energy_rate_90d_key")

    if st.button("1 Haftalık Tahmin Üret", type="primary", key="btn_reg"):
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
# TAB 2: SINIFLANDIRMA (aynı kalabilir)
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

    # Tab2 için: senin önceki compute_roll30_features fonksiyonun burada devam edebilir.
    st.info("Tab2 kısmı sende zaten çalışıyordu; istersen aynı mantıkla bırakalım.")

st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
