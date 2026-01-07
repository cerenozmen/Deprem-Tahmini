import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Sayfa Ayarları
st.set_page_config(
    page_title="İstanbul Deprem Tahmin Modeli",
    page_icon="mag",
    layout="wide"
)

# --- MODELLERİN YÜKLENMESİ ---
@st.cache_resource
def load_models():
    try:
        # Not: Dosya isimleri önceki adımda verdiğim isimlerle aynı olmalıdır
        reg_model = joblib.load('rf_reg_deprem_buyukluk.joblib')
        clf_model = joblib.load('rf_clf_deprem_olasilik.joblib')
        return reg_model, clf_model
    except FileNotFoundError as e:
        st.error(f"Model dosyaları bulunamadı! Lütfen .joblib dosyalarının 'app.py' ile aynı klasörde olduğundan emin olun. Hata: {e}")
        return None, None

rf_reg, rf_clf = load_models()

# --- YARDIMCI FONKSİYONLAR ---
def derive_date_features(selected_date):
    """Seçilen tarihten modelin ihtiyaç duyduğu özellikleri çıkarır."""
    return {
        "month": selected_date.month,
        "dow": selected_date.weekday(), # Day of week (0=Pazartesi)
        "dayofyear": selected_date.timetuple().tm_yday
    }

# --- ARAYÜZ ---
st.title("🌍 İstanbul Deprem Analiz ve Tahmin Paneli")
st.markdown("Bu uygulama, makine öğrenmesi modelleri kullanarak deprem büyüklüğü tahmini ve risk analizi yapar.")

# Sekmeler
tab1, tab2 = st.tabs(["📉 Deprem Büyüklüğü Tahmini (Regresyon)", "⚠️ Bölgesel Risk Analizi (Sınıflandırma)"])

# ---------------------------------------------------------
# TAB 1: REGRESYON (BÜYÜKLÜK TAHMİNİ)
# ---------------------------------------------------------
with tab1:
    st.header("Senaryo Bazlı Büyüklük Tahmini")
    st.info("Aşağıdaki parametreleri girerek olası bir depremin tahmini büyüklüğünü (Magnitude) hesaplayın.")

    if rf_reg is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📍 Konum ve Zaman")
            input_lat = st.number_input("Enlem (Latitude)", value=41.000, format="%.3f")
            input_lon = st.number_input("Boylam (Longitude)", value=29.000, format="%.3f")
            input_depth = st.number_input("Derinlik (km)", value=10.0, min_value=0.0)
            input_date = st.date_input("Tarih", datetime.date.today())
            
        with col2:
            st.subheader("🌋 Sismik Parametreler")
            input_fault_dist = st.number_input("Fay Hattına Uzaklık (km)", value=5.0)
            input_b_value = st.number_input("b-değeri (Sismik aktivite eğimi)", value=1.0)
            input_log_energy = st.number_input("Log Enerji", value=9.0)

        with col3:
            st.subheader("⚡ Enerji İstatistikleri (Detay)")
            # Kullanıcı kolaylığı için varsayılan ortalama değerler verildi
            input_e30 = st.number_input("30 Günlük Enerji", value=10000.0)
            input_e90 = st.number_input("90 Günlük Enerji", value=50000.0)
            # Diğer karmaşık feature'ları basitleştirmek için hidden calculation yapılabilir
            # Ancak model tam input beklediği için burada manuel giriş veya varsayılan bırakıyoruz
            with st.expander("Gelişmiş Enerji Parametreleri"):
                input_er30 = st.number_input("Enerji Hızı (30 Gün)", value=100.0)
                input_er90 = st.number_input("Enerji Hızı (90 Gün)", value=100.0)
                
                # Logaritmik değerleri otomatize edebiliriz
                input_log_e30 = np.log1p(input_e30)
                input_log_e90 = np.log1p(input_e90)
                input_log_er30 = np.log1p(input_er30)
                input_log_er90 = np.log1p(input_er90)

        # Tahmin Butonu
        if st.button("Büyüklüğü Tahmin Et", type="primary"):
            # Tarih özelliklerini çıkar
            date_feats = derive_date_features(input_date)
            
            # Modelin beklediği özellik sırasına göre DataFrame oluşturma
            # NOT: Bu isimler notebook'taki 'feature_cols' ile BİREBİR aynı olmalıdır.
            input_data = pd.DataFrame([{
                "lat": input_lat,
                "lon": input_lon,
                "depth_km": input_depth,
                "fault_distance": input_fault_dist,
                "b_value": input_b_value,
                "log_energy": input_log_energy,
                "energy_30d": input_e30,
                "energy_rate_30d": input_er30,
                "energy_90d": input_e90,
                "energy_rate_90d": input_er90,
                "log_energy_30d": input_log_e30,
                "log_energy_90d": input_log_e90,
                "log_energy_rate_30d": input_log_er30,
                "log_energy_rate_90d": input_log_er90,
                "month": date_feats['month'],
                "dow": date_feats['dow'],
                "dayofyear": date_feats['dayofyear']
            }])

            try:
                prediction = rf_reg.predict(input_data)[0]
                
                st.success("Tahmin Başarıyla Tamamlandı!")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Tahmini Büyüklük (Mw)", value=f"{prediction:.2f}")
                with metric_col2:
                    if prediction >= 7.0:
                        st.error("Durum: KRİTİK / YIKICI")
                    elif prediction >= 5.0:
                        st.warning("Durum: CİDDİ / ORTA")
                    else:
                        st.success("Durum: HAFİF / DÜŞÜK")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.write("Lütfen modelin feature sıralamasının kod ile eşleştiğinden emin olun.")

# ---------------------------------------------------------
# TAB 2: SINIFLANDIRMA (RİSK ANALİZİ)
# ---------------------------------------------------------
with tab2:
    st.header("Bölgesel Deprem Olasılığı (M ≥ 3.0)")
    st.write("Seçilen bölge ve geçmiş aktivite verilerine göre deprem olma olasılığını hesaplar.")
    
    if rf_clf is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Konum Bilgileri")
            # Model bin (kutu) mantığıyla çalıştığı için kullanıcıdan lat/lon alıp bin'e çeviriyoruz
            c_lat = st.number_input("Enlem", value=41.0, key="c_lat")
            c_lon = st.number_input("Boylam", value=29.0, key="c_lon")
            
            # Bin size 0.1 olarak notebookta belirtilmişti
            lat_bin = np.floor(c_lat / 0.1) * 0.1
            lon_bin = np.floor(c_lon / 0.1) * 0.1
            
            st.write(f"Hesaplanan Hücre: {lat_bin:.1f}, {lon_bin:.1f}")
            
        with c2:
            st.subheader("Geçmiş 30 Günlük Aktivite")
            st.caption("Bu değerler normalde veri tabanından otomatik çekilir. Senaryo için manuel giriniz.")
            
            roll30_count = st.number_input("Son 30 gündeki deprem sayısı", value=5.0)
            roll30_maxmag = st.number_input("Son 30 gündeki maks. büyüklük", value=3.5)
            roll30_meanmag = st.number_input("Son 30 gündeki ort. büyüklük", value=2.5)
            roll30_depth = st.number_input("Son 30 gündeki ort. derinlik", value=10.0)
            
            # Enerji verileri (basitleştirilmiş defaultlar)
            roll30_energy = 1000.0
            roll30_energy_rate = 10.0

        if st.button("Risk Hesapla", type="primary"):
            c_date_input = st.date_input("Analiz Tarihi", datetime.date.today(), key="c_date")
            c_date_feats = derive_date_features(c_date_input)
            
            # Sınıflandırma modeli için input DataFrame
            clf_input = pd.DataFrame([{
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "roll30_count": roll30_count,
                "roll30_maxmag": roll30_maxmag,
                "roll30_meanmag": roll30_meanmag,
                "roll30_depth": roll30_depth,
                "roll30_energy_30d": roll30_energy,
                "roll30_energy_rate_30d": roll30_energy_rate,
                "month": c_date_feats['month'],
                "dow": c_date_feats['dow'],
                "dayofyear": c_date_feats['dayofyear']
            }])
            
            try:
                # Olasılık tahmini (1 sınıfı olma ihtimali)
                prob = rf_clf.predict_proba(clf_input)[0][1]
                
                st.divider()
                st.subheader(f"M ≥ 3.0 Deprem Olasılığı: %{prob*100:.2f}")
                
                # Görselleştirme (Progress Bar)
                st.progress(prob)
                
                if prob > 0.7:
                    st.error("Yüksek Risk!")
                elif prob > 0.4:
                    st.warning("Orta Risk")
                else:
                    st.success("Düşük Risk")
                    
            except Exception as e:
                st.error(f"Sınıflandırma hatası: {e}")

# Footer
st.markdown("---")
st.caption("Geliştirilen bu arayüz prototip amaçlıdır. TÜBİTAK projesi kapsamında kullanılamaz.")
