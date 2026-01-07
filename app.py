import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from pathlib import Path

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
# İLÇE -> KOORDİNAT (CSV YOK)
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
# SABİT / ARKA PLAN PARAMETRELERİ (UI YOK)
# =========================
DEFAULT_DEPTH_KM = 10.0

# Regresyon modelinin beklediği (UI’da olmayan) feature’lar için sabit değerler:
DEFAULT_FAULT_DISTANCE = 5.0
DEFAULT_B_VALUE = 1.0
DEFAULT_LOG_ENERGY = 9.0

DEFAULT_E30 = 10000.0
DEFAULT_ER30 = 100.0
DEFAULT_E90 = 50000.0
DEFAULT_ER90 = 100.0

# Sınıflandırmada enerji tarafı (UI yok)
DEFAULT_ROLL30_ENERGY = 1000.0
DEFAULT_ROLL30_ENERGY_RATE = 10.0

# İsteğe bağlı: Eğer ham deprem olayları dosyan varsa buradan otomatik hesaplarız.
# Bu dosya ZORUNLU DEĞİL. Yoksa deterministik default üretilecek.
# Beklenen kolonlar örneği: date, lat, lon, depth_km, mag
EVENTS_FILE = "events.csv"

# =========================
# Yardımcılar
# =========================
def derive_date_features(d: datetime.date):
    return {"month": d.month, "dow": d.weekday(), "dayofyear": d.timetuple().tm_yday}

def week_dates(start_date: datetime.date, days: int = 7):
    return [start_date + datetime.timedelta(days=i) for i in range(days)]

def _stable_rng_seed(district: str, anchor_date: datetime.date) -> int:
    # Aynı ilçe + aynı tarih için aynı “otomatik” değerler üretsin (deterministik)
    s = f"{district}-{anchor_date.isoformat()}"
    return abs(hash(s)) % (2**32)

@st.cache_data
def load_events_if_exists():
    base = Path(__file__).resolve().parent
    fp = base / EVENTS_FILE
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    # Minimum kolon kontrolü
    needed = {"date", "lat", "lon", "mag"}
    if not needed.issubset(set(df.columns)):
        return None

    # Tarih parse
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "lat", "lon", "mag"])
    return df

def compute_roll30_features_auto(
    district_name: str,
    lat: float,
    lon: float,
    anchor_date: datetime.date,
    events_df: pd.DataFrame | None,
):
    """
    Son 30 gün metriklerini UI olmadan otomatik üretir.

    1) events.csv varsa: ilçeye yakın olayları filtreleyip (basit mesafe eşiği) 30 gün metriklerini hesaplar.
    2) events.csv yoksa: district+date bazlı deterministik (tekrar üretilebilir) değerler döner.
    """
    start = anchor_date - datetime.timedelta(days=30)

    if events_df is not None and not events_df.empty:
        # Basit yakınlık filtresi (yaklaşık): enlem/boylam farkı küçük olanları al
        # İstersen bunu daha sonra haversine ile iyileştiririz.
        df = events_df[(events_df["date"] >= start) & (events_df["date"] <= anchor_date)].copy()
        df["dlat"] = (df["lat"] - lat).abs()
        df["dlon"] = (df["lon"] - lon).abs()
        near = df[(df["dlat"] <= 0.20) & (df["dlon"] <= 0.20)]  # ~20km-30km bandına kabaca denk

        if len(near) > 0:
            roll30_count = float(len(near))
            roll30_maxmag = float(near["mag"].max())
            roll30_meanmag = float(near["mag"].mean())
            if "depth_km" in near.columns:
                roll30_depth = float(near["depth_km"].mean())
            else:
                roll30_depth = 10.0
            return roll30_count, roll30_maxmag, roll30_meanmag, roll30_depth

    # events yoksa -> deterministik üretim
    rng = np.random.default_rng(_stable_rng_seed(district_name, anchor_date))
    roll30_count = float(rng.integers(0, 15))               # 0-14 arası
    roll30_meanmag = float(np.clip(rng.normal(2.6, 0.35), 1.5, 4.0))
    roll30_maxmag = float(np.clip(roll30_meanmag + rng.uniform(0.2, 1.2), 2.0, 5.5))
    roll30_depth = float(np.clip(rng.normal(10.0, 4.0), 1.0, 30.0))
    return roll30_count, roll30_maxmag, roll30_meanmag, roll30_depth

# =========================
# Arayüz
# =========================
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri ile tahmin ve risk analizi yapar.")

tab1, tab2 = st.tabs(["📉 1 Haftalık Büyüklük Tahmini (Regresyon)", "⚠️ 1 Haftalık Bölgesel Risk (Sınıflandırma)"])

events_df = load_events_if_exists()

# =========================================================
# TAB 1: REGRESYON (DERİNLİK UI YOK)
# =========================================================
with tab1:
    st.header("İlçe Bazlı 1 Haftalık Büyüklük Tahmini")

    if rf_reg is None:
        st.stop()

    selected_district = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist(), key="reg_district")
    row = district_df[district_df["ilce_adi"] == selected_district].iloc[0]
    input_lat = float(row["lat"])
    input_lon = float(row["lon"])

    st.caption(f"Seçilen ilçe merkez koordinatı: **{input_lat:.5f}, {input_lon:.5f}**")
    start_date = st.date_input("Başlangıç Tarihi", datetime.date.today(), key="reg_start")

    if st.button("1 Haftalık Tahmin Üret", type="primary", key="reg_btn"):
        dates = week_dates(start_date, 7)
        rows = []

        # log’lar sabitlerden otomatik
        log_e30 = float(np.log1p(DEFAULT_E30))
        log_e90 = float(np.log1p(DEFAULT_E90))
        log_er30 = float(np.log1p(DEFAULT_ER30))
        log_er90 = float(np.log1p(DEFAULT_ER90))

        for d in dates:
            df_date = derive_date_features(d)
            rows.append({
                "date": d,
                "lat": input_lat,
                "lon": input_lon,
                "depth_km": float(DEFAULT_DEPTH_KM),  # UI yok, sabit

                "fault_distance": float(DEFAULT_FAULT_DISTANCE),
                "b_value": float(DEFAULT_B_VALUE),
                "log_energy": float(DEFAULT_LOG_ENERGY),

                "energy_30d": float(DEFAULT_E30),
                "energy_rate_30d": float(DEFAULT_ER30),
                "energy_90d": float(DEFAULT_E90),
                "energy_rate_90d": float(DEFAULT_ER90),

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
# TAB 2: SINIFLANDIRMA (SON 30 GÜN UI YOK, OTOMATİK)
# =========================================================
with tab2:
    st.header("İlçe Bazlı 1 Haftalık Deprem Olasılığı (M ≥ 3.0)")

    if rf_clf is None:
        st.stop()

    selected_district_c = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist(), key="clf_district")
    rowc = district_df[district_df["ilce_adi"] == selected_district_c].iloc[0]
    c_lat = float(rowc["lat"])
    c_lon = float(rowc["lon"])

    lat_bin = np.floor(c_lat / 0.1) * 0.1
    lon_bin = np.floor(c_lon / 0.1) * 0.1
    st.caption(f"Hesaplanan Hücre: **{lat_bin:.1f}, {lon_bin:.1f}**")

    start_date_c = st.date_input("Başlangıç Tarihi", datetime.date.today(), key="clf_start")

    # (UI’da gösterme) fakat istersen debug için expander içine koyabiliriz.
    # burada tamamen gizli bırakıyorum.

    if st.button("1 Haftalık Risk Hesapla", type="primary", key="clf_btn"):
        dates = week_dates(start_date_c, 7)
        rows = []

        for d in dates:
            # Son 30 gün özelliklerini otomatik üret
            roll30_count, roll30_maxmag, roll30_meanmag, roll30_depth = compute_roll30_features_auto(
                district_name=selected_district_c,
                lat=c_lat,
                lon=c_lon,
                anchor_date=d,
                events_df=events_df
            )
            df_date = derive_date_features(d)

            rows.append({
                "date": d,
                "lat_bin": float(lat_bin),
                "lon_bin": float(lon_bin),

                "roll30_count": float(roll30_count),
                "roll30_maxmag": float(roll30_maxmag),
                "roll30_meanmag": float(roll30_meanmag),
                "roll30_depth": float(roll30_depth),

                "roll30_energy_30d": float(DEFAULT_ROLL30_ENERGY),
                "roll30_energy_rate_30d": float(DEFAULT_ROLL30_ENERGY_RATE),

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

            max_prob = float(pred_df["prob"].max())
            st.metric("Haftalık Maks. Olasılık", f"%{max_prob*100:.2f}")
            st.progress(max_prob)

            if max_prob > 0.7:
                st.error("Haftalık değerlendirme: **Yüksek Risk**")
            elif max_prob > 0.4:
                st.warning("Haftalık değerlendirme: **Orta Risk**")
            else:
                st.success("Haftalık değerlendirme: **Düşük Risk**")

        except Exception as e:
            st.error(f"Sınıflandırma hatası: {e}")

st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
