# features_bandpower.py

import csv
import os
import sys

import numpy as np

import config


# =========================================================
# 1) BASİT AYAR
# =========================================================

# Session bilgisini artık sabit vermek yerine terminalden alacağız.
# Böylece aynı kodu bütün session'lar için tekrar tekrar kullanabiliriz.
# Örnek:
# python features_bandpower.py 1
# python features_bandpower.py 2

# Küçük bir epsilon:
# log(0) hatası olmasın diye kullanacağız
EPSILON = 1e-10


# =========================================================
# 2) WINDOW VERİSİNİ OKU
# =========================================================

def get_session_from_cli():
    """
    Terminalden session numarasını alır.

    Örnek:
    python features_bandpower.py 1
    """
    if len(sys.argv) < 2:
        raise ValueError("Kullanım: python features_bandpower.py <session_id>")

    session_name = str(sys.argv[1])
    return session_name


def load_window_data(session_name): #NPZ dosyası, Python'da NumPy kütüphanesi kullanılarak birden fazla diziyi veya matrisi tek bir sıkıştırılmış arşiv dosyasında saklamaya yarayan ikili (binary) bir dosya formatıdır.
    """
    windowing.py çıktısı olan npz dosyasını okur.

    Beklenen içerik:
    - X -> shape: (n_windows, n_channels, n_samples)
    - y -> shape: (n_windows,)
    """
    file_path = os.path.join(
        config.OUTPUT_DIR,
        "window_data",
        f"session_{session_name}_windows.npz"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Window verisi bulunamadı: {file_path}")

    data = np.load(file_path)

    X = data["X"]
    y = data["y"]

    return X, y


def load_window_metadata(session_name):
    """
    windowing.py'nin ürettiği metadata CSV dosyasını okur.
    """
    file_path = os.path.join(
        config.OUTPUT_DIR,
        "window_data",
        f"session_{session_name}_window_metadata.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Window metadata bulunamadı: {file_path}")

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


# =========================================================
# 3) BANDPOWER ÖZELLİĞİ
# =========================================================

def extract_bandpower_features(X):
    """
    Her pencere için kanal bazlı bandpower çıkarır.

    Mantık:
    - X zaten 8-30 Hz filtrelenmiş pencere verisi
    - Bu yüzden her kanalda mean(signal^2) almak,
      o banttaki gücün sade bir özetidir.

    Girdi:
        X -> (n_windows, n_channels, n_samples)

    Çıktı:
        features -> (n_windows, n_channels)
    """
    # Kare al, zaman boyunca ortalama al
    power = np.mean(X ** 2, axis=2)

    # İleride EEG'de pratik kullanım olan log bandpower'a geçebiliriz. Log bandpower, EEG gücünün logaritmasını alarak daha stabil ve modelin öğrenmesi için daha uygun
    # özellikler elde etmemizi sağlar. Log dönüşümü, özellikle EEG gücü gibi genellikle log-normal dağılıma sahip verilerde,
    # özelliklerin dağılımını normalize eder ve aşırı büyük değerlerin etkisini azaltır.
    # Pratik olması için küçük bir epsilon ekleyelim, böylece log(0) hatası almayız. Bu, özellikle bazı pencerelerde gücün çok düşük olduğu durumlarda önemlidir.

    # Daha stabil ve yaygın kullanım için log uygula
    log_power = np.log(power + EPSILON)

    return log_power


# =========================================================
# 4) FEATURE NAME ÜRET
# =========================================================

def build_feature_names(n_channels):
    """
    Özellik isimlerini üretir.

    Şimdilik her özellik:
    channel_i_8_30_logpower
    şeklinde adlandırılır.        *(GPT 5.4 tarafından önerildi)*
    """
    feature_names = []

    for i in range(n_channels):
        feature_names.append(f"channel_{i+1}_8_30_logpower")

    return feature_names


# =========================================================
# 5) KISA KONTROL
# =========================================================

def print_feature_summary(X, y, features):
    """
    Özelliklerin temel özetini ekrana basar.
    """
    print("\n===== FEATURE ÖZETİ =====")
    print("Orijinal window shape:", X.shape)
    print("Feature matrix shape:", features.shape)
    print("Label shape:", y.shape)
    print("Label 0 sayısı:", int(np.sum(y == 0)))
    print("Label 1 sayısı:", int(np.sum(y == 1)))


def print_first_feature_rows(features, y, max_rows=5):
    """
    İlk birkaç feature satırını gösterir.
    """
    print("\n===== İLK FEATURE SATIRLARI =====")

    limit = min(max_rows, len(features))

    for i in range(limit):
        first_values = np.round(features[i][:5], 4)  # ilk 5 feature
        print(
            f"window_index={i} | "
            f"label={int(y[i])} | "
            f"ilk_5_feature={first_values}"
        )


# =========================================================
# 6) KAYDETME
# =========================================================

def save_feature_npz(features, y, session_name): #bu fonskyion, özellik matrisini ve label vektörünü sıkıştırılmış NPZ formatında kaydeder. NPZ formatı, NumPy dizilerini tek bir dosyada saklamak için kullanılır ve genellikle büyük veri setleri için tercih edilir. Bu fonksiyon, features ve y dizilerini session_name ile adlandırılmış bir NPZ dosyasına kaydeder. Kaydedilen dosya, daha sonra model eğitimi veya analiz için kolayca yüklenebilir.
    """
    Özellik matrisi ve label vektörünü kaydeder.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "features")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_bandpower_features.npz")

    np.savez_compressed(save_path, X_features=features, y=y)

    print(f"\nFeature verisi kaydedildi: {save_path}")


def save_feature_names_csv(feature_names, session_name): #bu fonksiyon, özellik isimlerini CSV formatında kaydeder. Özellik isimleri, model eğitimi ve yorumlama sürecinde önemlidir çünkü hangi özelliğin ne anlama geldiğini bilmek, modelin nasıl çalıştığını anlamamıza yardımcı olur. Bu fonksiyon, feature_names listesini session_name ile adlandırılmış bir CSV dosyasına kaydeder. Kaydedilen dosya, daha sonra model değerlendirme veya sonuç raporlama süreçlerinde kullanılabilir.
    """
    Özellik isimlerini CSV olarak kaydeder.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "features")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_feature_names.csv")

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])

        for i, name in enumerate(feature_names):
            writer.writerow([i, name])

    print(f"Feature isimleri kaydedildi: {save_path}")


def save_feature_table_csv(features, y, feature_names, metadata_rows, session_name): 
    """
    Özellikleri daha kolay gözle kontrol etmek için CSV olarak da kaydeder.

    Not:
    Büyük dosya olabilir ama tek session için yönetilebilir. Subject'ler arttıkça bu dosyaların boyutu da artacak, ama şimdilik sorun olmaz. İleride gerekirse sadece sunum amaçlı bir kısmı kaydedilebilir.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "features")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_bandpower_features.csv")

    header = [
        "window_id",
        "trial_id",
        "session_id",
        "run_id",
        "source_segment",
        "label",
        "cue_label",
        "window_start_sec",
        "window_end_sec",
    ] + feature_names

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(features)):
            meta = metadata_rows[i]

            row = [
                meta["window_id"],
                meta["trial_id"],
                meta["session_id"],
                meta["run_id"],
                meta["source_segment"],
                int(y[i]),
                meta["cue_label"],
                meta["window_start_sec"],
                meta["window_end_sec"],
            ] + list(np.round(features[i], 6))

            writer.writerow(row)

    print(f"Feature tablosu kaydedildi: {save_path}")


# =========================================================
# 7) ANA AKIŞ
# =========================================================

def main():
    session_name = get_session_from_cli()

    print(f"Session {session_name} için window verisi okunuyor...")
    X, y = load_window_data(session_name)

    print("Window metadata okunuyor...")
    metadata_rows = load_window_metadata(session_name)

    if len(metadata_rows) != len(X):
        raise ValueError("Metadata satır sayısı ile pencere sayısı eşleşmiyor.")

    print("Bandpower feature çıkarılıyor...")
    features = extract_bandpower_features(X)

    n_channels = X.shape[1]
    feature_names = build_feature_names(n_channels)

    print_feature_summary(X, y, features)
    print_first_feature_rows(features, y)

    save_feature_npz(features, y, session_name)
    save_feature_names_csv(feature_names, session_name)
    save_feature_table_csv(features, y, feature_names, metadata_rows, session_name)

    print("\nBandpower feature çıkarımı tamamlandı.")


if __name__ == "__main__":
    main()