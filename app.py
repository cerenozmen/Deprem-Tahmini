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
@st.cache_data
def load_istanbul_districts():
    """
    ilce.csv dosyasından İstanbul ilçelerini çeker.
    Beklenen kolonlar: il_plaka, ilce_adi, lat, lon (gist formatı ile uyumlu)
    """
    try:
        df = pd.read_csv("ilce.csv", sep=None, engine="python")  # otomatik ayırıcı dene
    except FileNotFoundError:
        st.error(
            "ilce.csv bulunamadı! Lütfen ilce.csv dosyasını app.py ile aynı klasöre koyun.\n"
            "Dosya, Türkiye il/ilçe koordinatlarını içermeli (kolonlar: il_plaka, ilce_adi, lat, lon)."
        )
        return pd.DataFrame(columns=["il_plaka", "ilce_adi", "lat", "lon"])

    # Bazı dosyalarda kolon adı lon/lng olabilir
    if "lon" not in df.columns and "lng" in df.columns:
        df = df.rename(columns={"lng": "lon"})

    needed = {"il_plaka", "ilce_adi", "lat", "lon"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"ilce.csv eksik kolonlar var: {sorted(list(missing))}")
        return pd.DataFrame(columns=["il_plaka", "ilce_adi", "lat", "lon"])

    # İstanbul plaka = 34
    ist = df[df["il_plaka"].astype(str).str.strip() == "34"].copy()
    ist["ilce_adi"] = ist["ilce_adi"].astype(str).str.strip()

    # Aynı ilçe birden fazla kez geçiyorsa ilkini al
    ist = ist.drop_duplicates(subset=["ilce_adi"], keep="first").sort_values("ilce_adi")
    return ist[["ilce_adi", "lat", "lon"]].reset_index(drop=True)

district_df = load_istanbul_districts()

# =========================
# Yardımcılar
# =========================
def derive_date_features(d: datetime.date):
    return {
        "month": d.month,
        "dow": d.weekday(),  # 0=Pazartesi
        "dayofyear": d.timetuple().tm_yday
    }

def week_dates(start_date: datetime.date, days: int = 7):
    return [start_date + datetime.timedelta(days=i) for i in range(days)]

# =========================
# Arayüz
# =========================
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri ile tahmin ve risk analizi yapar.")

tab1, tab2 = st.tabs(["📉 1 Haftalık Büyüklük Tahmini (Regresyon)", "⚠️ 1 Haftalık Bölgesel Risk (Sınıflandırma)"])

# =========================================================
# TAB 1: REGRESYON (1 HAFTALIK)
# =========================================================
with tab1:
    st.header("İlçe Bazlı 1 Haftalık Büyüklük Tahmini")
    st.info("İlçe + derinlik + başlangıç tarihi seçerek sonraki 7 gün için tahmini Mw üretir.")

    if rf_reg is None:
        st.stop()

    if district_df.empty:
        st.warning("İlçe listesi yüklenemedi (ilce.csv). İlçe seçimi yapılamıyor.")
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("📍 İlçe ve Senaryo")
        selected_district = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist())
        row = district_df[district_df["ilce_adi"] == selected_district].iloc[0]
        input_lat = float(row["lat"])
        input_lon = float(row["lon"])

        st.caption(f"Seçilen ilçe merkez koordinatı: **{input_lat:.5f}, {input_lon:.5f}**")

        input_depth = st.number_input("Derinlik (km)", value=10.0, min_value=0.0)
        start_date = st.date_input("Başlangıç Tarihi", datetime.date.today())

    with c2:
        st.subheader("⚙️ Arka Plan Varsayılanları (UI’dan kaldırıldı)")
        st.caption("Modelin beklediği sismik/enerji feature’ları arayüzde yok; sabit değerlerle dolduruluyor.")
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

        # log’ları otomatik üret
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
                "depth_km": input_depth,

                # UI’dan kaldırılanlar -> default
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

        # Model inputu: date hariç
        model_input = pred_df.drop(columns=["date"]).copy()

        try:
            preds = rf_reg.predict(model_input)
            pred_df["pred_mw"] = preds

            st.success("1 haftalık tahmin üretildi ✅")

            show_df = pred_df[["date", "pred_mw"]].copy()
            show_df["pred_mw"] = show_df["pred_mw"].astype(float).round(2)
            st.dataframe(show_df, use_container_width=True)

            max_pred = float(pred_df["pred_mw"].max())
            st.metric("Haftalık Maks. Tahmini Mw", f"{max_pred:.2f}")

            if max_pred >= 7.0:
                st.error("Haftalık değerlendirme: **KRİTİK / YIKICI**")
            elif max_pred >= 5.0:
                st.warning("Haftalık değerlendirme: **CİDDİ / ORTA**")
            else:
                st.success("Haftalık değerlendirme: **HAFİF / DÜŞÜK**")

        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            st.write("İpucu: Modelin beklediği feature isim/sırası ile bu kodun birebir uyuştuğundan emin olun.")

# =========================================================
# TAB 2: SINIFLANDIRMA (1 HAFTALIK)
# =========================================================
with tab2:
    st.header("İlçe Bazlı 1 Haftalık Deprem Olasılığı (M ≥ 3.0)")
    st.write("İlçe + son 30 gün aktivitesi + başlangıç tarihi ile 7 günlük olasılık üretir.")

    if rf_clf is None:
        st.stop()

    if district_df.empty:
        st.warning("İlçe listesi yüklenemedi (ilce.csv). İlçe seçimi yapılamıyor.")
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("📍 İlçe")
        selected_district_c = st.selectbox("İlçe seçin", district_df["ilce_adi"].tolist(), key="district_clf")
        rowc = district_df[district_df["ilce_adi"] == selected_district_c].iloc[0]
        c_lat = float(rowc["lat"])
        c_lon = float(rowc["lon"])

        # Modeliniz bin mantığı ile çalışıyorsa:
        lat_bin = np.floor(c_lat / 0.1) * 0.1
        lon_bin = np.floor(c_lon / 0.1) * 0.1
        st.caption(f"Hesaplanan Hücre: **{lat_bin:.1f}, {lon_bin:.1f}**")

        start_date_c = st.date_input("Başlangıç Tarihi", datetime.date.today(), key="start_date_c")

    with c2:
        st.subheader("📊 Son 30 Gün Aktivitesi (Basit)")
        st.caption("Enerji parametreleri UI’dan kaldırıldı; arkada default dolduruluyor.")
        roll30_count = st.number_input("Son 30 gündeki deprem sayısı", value=5.0)
        roll30_maxmag = st.number_input("Son 30 gündeki maks. büyüklük", value=3.5)
        roll30_meanmag = st.number_input("Son 30 gündeki ort. büyüklük", value=2.5)
        roll30_depth = st.number_input("Son 30 gündeki ort. derinlik", value=10.0)

        with st.expander("Arka plan enerji defaultları (opsiyonel)"):
            roll30_energy = st.number_input("roll30_energy_30d [default]", value=1000.0)
            roll30_energy_rate = st.number_input("roll30_energy_rate_30d [default]", value=10.0)

    if st.button("1 Haftalık Risk Hesapla", type="primary"):
        dates = week_dates(start_date_c, 7)
        rows = []

        for d in dates:
            df_date = derive_date_features(d)
            rows.append({
                "date": d,
                "lat_bin": float(lat_bin),
                "lon_bin": float(lon_bin),
                "roll30_count": float(roll30_count),
                "roll30_maxmag": float(roll30_maxmag),
                "roll30_meanmag": float(roll30_meanmag),
                "roll30_depth": float(roll30_depth),

                # UI’dan kaldırıldı -> default
                "roll30_energy_30d": float(roll30_energy),
                "roll30_energy_rate_30d": float(roll30_energy_rate),

                "month": df_date["month"],
                "dow": df_date["dow"],
                "dayofyear": df_date["dayofyear"]
            })

        pred_df = pd.DataFrame(rows)
        model_input = pred_df.drop(columns=["date"]).copy()

        try:
            probs = rf_clf.predict_proba(model_input)[:, 1]
            pred_df["prob"] = probs

            st.success("1 haftalık risk üretildi ✅")

            show_df = pred_df[["date", "prob"]].copy()
            show_df["prob_%"] = (show_df["prob"] * 100).round(2)
            show_df = show_df.drop(columns=["prob"])
            st.dataframe(show_df, use_container_width=True)

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

# Footer
st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
