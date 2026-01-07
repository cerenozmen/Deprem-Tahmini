import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math
import os

# =========================
# Sayfa Ayarları
# =========================
st.set_page_config(
    page_title="İstanbul Deprem Tahmin Modeli",
    page_icon="🌍",
    layout="wide"
)

# =========================
# VERİ YÜKLEME (CSV'DEN)
# =========================
@st.cache_data
def load_quake_catalog_from_file(file_path):
    """
    Dışarıdaki bir CSV dosyasından deprem verilerini yükler.
    Dosya yoksa boş bir DataFrame döner.
    """
    if not os.path.exists(file_path):
        st.error(f"⚠️ HATA: '{file_path}' dosyası bulunamadı! Lütfen veri setini klasöre ekleyin.")
        return pd.DataFrame(columns=["time", "lat", "lon", "mag", "depth_km"])
    
    try:
        df = pd.read_csv(file_path)
        # Tarih sütununu dönüştür
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # Eksik verileri temizle
        df = df.dropna(subset=["time", "lat", "lon", "mag"]).copy()
        # Tip dönüşümleri
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        df["mag"] = df["mag"].astype(float)
        # Derinlik yoksa varsayılan 10km ata
        if "depth_km" not in df.columns:
            df["depth_km"] = 10.0
        else:
            df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce").fillna(10.0)
            
        return df
    except Exception as e:
        st.error(f"Veri okunurken hata oluştu: {e}")
        return pd.DataFrame()

# Katalogu yükle (Dosya adını buradan değiştirebilirsin)
quake_catalog = load_quake_catalog_from_file("deprem_verileri.csv")

# =========================
# MODELLER
# =========================
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load("rf_reg_deprem_buyukluk.joblib")
        clf_model = joblib.load("rf_clf_deprem_olasilik.joblib")
        return reg_model, clf_model
    except Exception as e:
        st.error(f"Model dosyaları (.joblib) yüklenemedi: {e}")
        return None, None

rf_reg, rf_clf = load_models()

# =========================
# İLÇE -> KOORDİNAT (SABİT)
# =========================
DISTRICTS = {
    "Adalar": (40.8739, 29.1236), "Arnavutköy": (41.1831, 28.7406), "Ataşehir": (40.9923, 29.1244),
    "Avcılar": (40.9792, 28.7214), "Bağcılar": (41.0390, 28.8564), "Bahçelievler": (40.9967, 28.8500),
    "Bakırköy": (40.9833, 28.8725), "Başakşehir": (41.0937, 28.8020), "Bayrampaşa": (41.0466, 28.9023),
    "Beşiktaş": (41.0422, 29.0094), "Beykoz": (41.1340, 29.0950), "Beylikdüzü": (41.0015, 28.6417),
    "Beyoğlu": (41.0369, 28.9847), "Büyükçekmece": (41.0205, 28.5850), "Çatalca": (41.1429, 28.4610),
    "Çekmeköy": (41.0352, 29.1757), "Esenler": (41.0437, 28.8762), "Esenyurt": (41.0343, 28.6801),
    "Eyüpsultan": (41.0480, 28.9330), "Fatih": (41.0186, 28.9390), "Gaziosmanpaşa": (41.0584, 28.9153),
    "Güngören": (41.0179, 28.8790), "Kadıköy": (40.9917, 29.0275), "Kağıthane": (41.0853, 28.9780),
    "Kartal": (40.9006, 29.1894), "Küçükçekmece": (41.0009, 28.7906), "Maltepe": (40.9357, 29.1551),
    "Pendik": (40.8775, 29.2356), "Sancaktepe": (41.0090, 29.2130), "Sarıyer": (41.1667, 29.0500),
    "Silivri": (41.0744, 28.2464), "Sultanbeyli": (40.9680, 29.2690), "Sultangazi": (41.1065, 28.8687),
    "Şile": (41.1755, 29.6130), "Şişli": (41.0602, 28.9877), "Tuzla": (40.8183, 29.3006),
    "Ümraniye": (41.0164, 29.1240), "Üsküdar": (41.0220, 29.0320), "Zeytinburnu": (40.9941, 28.9033),
}

district_df = pd.DataFrame(
    [{"ilce_adi": k, "lat": v[0], "lon": v[1]} for k, v in DISTRICTS.items()]
).sort_values("ilce_adi").reset_index(drop=True)

# Fay Hattı Tanımı
FAULT_POINTS = [(40.75, 28.20), (40.75, 28.60), (40.78, 29.00), (40.80, 29.40)]

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def mag_to_energy(m):
    return float(10 ** (1.5 * m + 4.8))

def derive_date_features(d):
    return {"month": d.month, "dow": d.weekday(), "dayofyear": d.timetuple().tm_yday}

def fault_distance_km(lat, lon):
    return float(min(haversine_km(lat, lon, fp[0], fp[1]) for fp in FAULT_POINTS))

def b_value_mle(mags, mmin=0.0):
    mags = np.asarray(mags, dtype=float)
    mags = mags[mags >= mmin]
    if mags.size < 2: return 1.0
    denom = np.mean(mags) - (mmin - 0.05)
    return float(np.log10(np.e) / denom) if denom > 0 else 1.0

def window_events(df, center_lat, center_lon, start_dt, end_dt, radius_km=30.0):
    sub = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].copy()
    if sub.empty: return sub
    sub["_dist_km"] = sub.apply(lambda r: haversine_km(center_lat, center_lon, r["lat"], r["lon"]), axis=1)
    return sub[sub["_dist_km"] <= radius_km]

def summarize_roll30(sub):
    if sub.empty:
        return {"roll30_count": 0.0, "roll30_maxmag": 0.0, "roll30_meanmag": 0.0, "roll30_depth": 0.0, "roll30_energy_30d": 0.0, "roll30_energy_rate_30d": 0.0}
    energies = sub["mag"].apply(mag_to_energy)
    return {
        "roll30_count": float(len(sub)),
        "roll30_maxmag": float(sub["mag"].max()),
        "roll30_meanmag": float(sub["mag"].mean()),
        "roll30_depth": float(sub["depth_km"].mean()),
        "roll30_energy_30d": float(energies.sum()),
        "roll30_energy_rate_30d": float(energies.sum() / 30.0)
    }

def compute_reg_features_from_dataset(quake_df, center_lat, center_lon, as_of_date):
    day0 = datetime.datetime.combine(as_of_date, datetime.time.min)
    sub30 = window_events(quake_df, center_lat, center_lon, day0 - datetime.timedelta(days=30), day0)
    sub90 = window_events(quake_df, center_lat, center_lon, day0 - datetime.timedelta(days=90), day0)
    
    e30 = sub30["mag"].apply(mag_to_energy).sum() if not sub30.empty else 0.0
    e90 = sub90["mag"].apply(mag_to_energy).sum() if not sub90.empty else 0.0
    
    return {
        "fault_distance": fault_distance_km(center_lat, center_lon),
        "b_value": b_value_mle(sub30["mag"].values) if not sub30.empty else 1.0,
        "log_energy": float(np.log1p(e30)),
        "energy_30d": float(e30),
        "energy_rate_30d": float(e30 / 30.0),
        "energy_90d": float(e90),
        "energy_rate_90d": float(e90 / 90.0),
    }

def compute_roll30_features(quake_df, lat_bin, lon_bin, as_of_date, center_lat, center_lon):
    start_dt = datetime.datetime.combine(as_of_date - datetime.timedelta(days=30), datetime.time.min)
    end_dt = datetime.datetime.combine(as_of_date, datetime.time.min)
    dfw = quake_df[(quake_df["time"] >= start_dt) & (quake_df["time"] < end_dt)].copy()
    
    # Hücre bazlı filtreleme
    sub = dfw[(dfw["lat"] >= lat_bin) & (dfw["lat"] < lat_bin + 0.1) & (dfw["lon"] >= lon_bin) & (dfw["lon"] < lon_bin + 0.1)]
    
    if not sub.empty:
        res = summarize_roll30(sub)
        res["_source"] = "cell"
        return res
    
    # Fallback: Yarıçap (30km)
    dfw["_dist"] = dfw.apply(lambda r: haversine_km(center_lat, center_lon, r["lat"], r["lon"]), axis=1)
    sub_r = dfw[dfw["_dist"] <= 30.0]
    res = summarize_roll30(sub_r)
    res["_source"] = "radius_30km"
    return res

# =========================
# ARAYÜZ (STREAMLIT)
# =========================
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.info(f"Katalogda şu an **{len(quake_catalog)}** kayıt yüklü.")

tab1, tab2 = st.tabs(["📉 1 Haftalık Büyüklük Tahmini", "⚠️ 1 Haftalık Bölgesel Risk"])

# TAB 1: REGRESYON
with tab1:
    if rf_reg:
        c1, c2 = st.columns(2)
        with c1:
            selected_district = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist())
            row = district_df[district_df["ilce_adi"] == selected_district].iloc[0]
            start_date = st.date_input("Başlangıç Tarihi", datetime.date.today())
            
            auto_reg = compute_reg_features_from_dataset(quake_catalog, row["lat"], row["lon"], start_date)

        with c2:
            st.subheader("⚙️ Otomatik Hesaplanan Parametreler")
            st.json(auto_reg)

        if st.button("7 Günlük Tahmini Başlat"):
            preds = []
            for i in range(7):
                d = start_date + datetime.timedelta(days=i)
                feats = derive_date_features(d)
                # Modelin beklediği tüm feature'ları buraya ekle (Eğitim sırasıyla aynı olmalı)
                input_data = pd.DataFrame([{
                    "lat": row["lat"], "lon": row["lon"], "depth_km": 10.0,
                    "fault_distance": auto_reg["fault_distance"], "b_value": auto_reg["b_value"],
                    "log_energy": auto_reg["log_energy"], "energy_30d": auto_reg["energy_30d"],
                    "energy_rate_30d": auto_reg["energy_rate_30d"], "energy_90d": auto_reg["energy_90d"],
                    "energy_rate_90d": auto_reg["energy_rate_90d"],
                    "log_energy_30d": np.log1p(auto_reg["energy_30d"]), "log_energy_90d": np.log1p(auto_reg["energy_90d"]),
                    "log_energy_rate_30d": np.log1p(auto_reg["energy_rate_30d"]), "log_energy_rate_90d": np.log1p(auto_reg["energy_rate_90d"]),
                    "month": feats["month"], "dow": feats["dow"], "dayofyear": feats["dayofyear"]
                }])
                val = rf_reg.predict(input_data)[0]
                preds.append({"Tarih": d, "Tahmin Edilen Büyüklük (Mw)": round(val, 2)})
            
            st.table(pd.DataFrame(preds))

# TAB 2: SINIFLANDIRMA
with tab2:
    if rf_clf:
        c1, c2 = st.columns(2)
        with c1:
            selected_district_c = st.selectbox("İlçe seçin (Risk)", district_df["ilce_adi"].tolist())
            rowc = district_df[district_df["ilce_adi"] == selected_district_c].iloc[0]
            lat_bin, lon_bin = math.floor(rowc["lat"]/0.1)*0.1, math.floor(rowc["lon"]/0.1)*0.1
            start_date_c = st.date_input("Başlangıç Tarihi (Risk)", datetime.date.today())

        auto_clf = compute_roll30_features(quake_catalog, lat_bin, lon_bin, start_date_c, rowc["lat"], rowc["lon"])

        with c2:
            st.subheader("📊 Sismik Aktivite Verileri (Son 30 Gün)")
            st.write(f"Kaynak: {auto_clf.get('_source')}")
            st.write(f"Deprem Sayısı: {auto_clf['roll30_count']}")
            st.write(f"Maks Büyüklük: {auto_clf['roll30_maxmag']}")

        if st.button("Risk Analizi Yap"):
            risk_results = []
            for i in range(7):
                d = start_date_c + datetime.timedelta(days=i)
                df = derive_date_features(d)
                input_clf = pd.DataFrame([{
                    "lat_bin": lat_bin, "lon_bin": lon_bin,
                    "roll30_count": auto_clf["roll30_count"], "roll30_maxmag": auto_clf["roll30_maxmag"],
                    "roll30_meanmag": auto_clf["roll30_meanmag"], "roll30_depth": auto_clf["roll30_depth"],
                    "roll30_energy_30d": auto_clf["roll30_energy_30d"], "roll30_energy_rate_30d": auto_clf["roll30_energy_rate_30d"],
                    "month": df["month"], "dow": df["dow"], "dayofyear": df["dayofyear"]
                }])
                prob = rf_clf.predict_proba(input_clf)[0][1]
                risk_results.append({"Tarih": d, "M≥3.0 Olasılığı (%)": round(prob * 100, 2)})
            
            st.dataframe(pd.DataFrame(risk_results), use_container_width=True)

st.divider()
st.caption("Prototip amaçlıdır. Veri seti doğruluğu modele bağlıdır.")
