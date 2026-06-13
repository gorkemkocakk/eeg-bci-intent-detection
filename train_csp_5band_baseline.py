# train_csp_5band_baseline.py

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config


# =========================================================
# 1) KUCUK YARDIMCI FONKSIYON
# =========================================================

def sort_session_ids(session_ids):
    """
    Session isimlerini sayisal siraya gore siralar.
    """
    return sorted(session_ids, key=lambda x: int(x))


def resolve_csp_components(csp_components=None):
    """
    Kullanılacak CSP component sayısını belirler ve doğrular.
    """
    if csp_components is None:
        value = int(config.CSP_COMPONENTS)
    else:
        value = int(csp_components)

    if value <= 0:
        raise ValueError("csp_components pozitif bir tamsayı olmalı.")

    return value


def get_canonical_bands():
    """
    Canonical 5 bandi sabit sirada dondurur.
    """
    # Bant sirasi feature kolon sirasi haline gelir.
    # Bu nedenle ayni sira egitim, raporlama ve XAI ciktisinda korunmalidir.
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    bands = []

    for name in band_names:
        if name not in config.CANONICAL_BANDS:
            raise ValueError(f"Config icinde canonical band eksik: {name}")

        low_freq, high_freq = config.CANONICAL_BANDS[name]
        bands.append((name, float(low_freq), float(high_freq)))

    return bands


# =========================================================
# 2) LOSO-SESSION FOLD'LARI
# =========================================================

def build_loso_folds(session_data):
    """
    Leave-one-session-out fold listesini uretir.
    """
    session_ids = sort_session_ids(list(session_data.keys()))

    if len(session_ids) < 2:
        raise ValueError(
            "Cross-session 5-band CSP baseline icin en az 2 session window dosyasi gerekir."
        )

    folds = []

    # Session shuffle uygulanmaz; her session sirayla test oturumu olur.
    # LOSO-session, farkli oturuma genelleme performansini olcmek icin kullanilir.
    for test_session in session_ids:
        train_sessions = [s for s in session_ids if s != test_session]

        folds.append(
            {
                "test_session": test_session,
                "train_sessions": train_sessions
            }
        )

    return folds


# =========================================================
# 3) TRAIN / TEST MATRISLERINI OLUSTUR
# =========================================================

def build_train_test_data(session_data, fold):
    """
    Verilen fold icin train ve test matrislerini uretir.
    """
    X_train_list = []
    y_train_list = []

    for session_id in fold["train_sessions"]:
        X_train_list.append(session_data[session_id]["X"])
        y_train_list.append(session_data[session_id]["y"])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    test_session = fold["test_session"]
    X_test = session_data[test_session]["X"]
    y_test = session_data[test_session]["y"]

    return X_train, y_train, X_test, y_test


# =========================================================
# 4) 5-BAND CSP FEATURE CIKARIMI (TRAIN-ONLY FIT)
# =========================================================

def apply_bandpass_to_epochs(X, sfreq, low_freq, high_freq):
    """
    Epoch dizisine bandpass uygular.

    X shape: (n_epochs, n_channels, n_samples)
    """
    # Wideband pencere burada ilgili canonical banda ayrilir.
    # Bu filtre supervised degildir; sinif bilgisini kullanan asama CSP fit'idir.
    X_filtered = filter_data(
        X,
        sfreq=sfreq,
        l_freq=low_freq,
        h_freq=high_freq,
        method="iir",  # Kisa pencerelerde FIR filter-length warning'ini azaltmak icin
        iir_params={"order": 4, "ftype": "butter"},
        verbose=False
    )
    return X_filtered


def extract_log_variance(csp_space_data):
    """
    CSP uzayindaki zaman serilerinden log-variance feature uretir.
    """
    variance = np.var(csp_space_data, axis=2)
    log_variance = np.log(variance + 1e-10)
    return log_variance


def fit_transform_one_band_csp(X_train_band, y_train, X_test_band, csp_components=None):
    """
    Tek bir band icin CSP'yi sadece train veride fit eder.
    """
    csp_components_value = resolve_csp_components(csp_components)

    # Her bant kendi CSP filtresini train veriden ogrenir.
    # Test bandini fit'e katmak kovaryans bilgisini sizdirir ve skoru iyimserlestirir.
    csp_model = CSP(
        n_components=csp_components_value,
        reg="ledoit_wolf",  # Sayisal kararlilik
        log=None,
        transform_into="csp_space",
        norm_trace=False
    )

    X_train_csp = csp_model.fit_transform(X_train_band, y_train)
    X_test_csp = csp_model.transform(X_test_band)

    X_train_feat = extract_log_variance(X_train_csp)
    X_test_feat = extract_log_variance(X_test_csp)

    return X_train_feat, X_test_feat


def build_5band_csp_features(X_train, y_train, X_test, sfreq, csp_components=None):
    """
    Her bant icin ayri CSP uygular, sonra tum bant feature'larini birlestirir.
    """
    bands = get_canonical_bands()

    train_feature_blocks = []
    test_feature_blocks = []

    # Her bandin CSP feature blogu ayni sirada biriktirilir.
    # Son feature matrisi: [delta CSP, theta CSP, alpha CSP, beta CSP, gamma CSP] sirasindadir.
    for band_name, low_freq, high_freq in bands:
        X_train_band = apply_bandpass_to_epochs(X_train, sfreq, low_freq, high_freq)
        X_test_band = apply_bandpass_to_epochs(X_test, sfreq, low_freq, high_freq)

        X_train_feat, X_test_feat = fit_transform_one_band_csp(
            X_train_band,
            y_train,
            X_test_band,
            csp_components=csp_components
        )

        train_feature_blocks.append(X_train_feat)
        test_feature_blocks.append(X_test_feat)

    X_train_all = np.concatenate(train_feature_blocks, axis=1)
    X_test_all = np.concatenate(test_feature_blocks, axis=1)

    return X_train_all, X_test_all


# =========================================================
# 5) TEK FOLD CALISTIR
# =========================================================

def run_one_fold(X_train, y_train, X_test, y_test, csp_components=None):
    """
    Tek bir LOSO fold'unu calistirir.

    Kural:
    - Her bant CSP sadece train'de fit edilir
    - test sadece train-CSP ile transform edilir
    - scaler sadece train'de fit edilir
    - sonra LDA egitilir
    """
    csp_components_value = resolve_csp_components(csp_components)

    n_channels = int(X_train.shape[1])
    # CSP component sayisi kanal sayisini asarsa matematiksel olarak anlamli filtre uretilemez.
    # Bu kontrol ablation calismalarinda hatayi model fit asamasindan once yakalar.
    if csp_components_value > n_channels:
        raise ValueError(
            f"csp_components ({csp_components_value}) kanal sayısından büyük olamaz "
            f"(n_channels={n_channels})."
        )

    X_train_feat, X_test_feat = build_5band_csp_features(
        X_train,
        y_train,
        X_test,
        sfreq=float(config.TARGET_SFREQ),
        csp_components=csp_components_value
    )

    # 5-band CSP'den sonra tum feature'lar ayni olcekte olmayabilir.
    # StandardScaler yalnizca train feature'larinda fit edilerek test session disarida tutulur.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train_scaled, y_train)

    y_pred = lda_model.predict(X_test_scaled)

    if hasattr(lda_model, "predict_proba"):
        y_score = lda_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = lda_model.decision_function(X_test_scaled)

    auc = roc_auc_score(y_test, y_score)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return auc, bal_acc, cm, y_pred, y_score


# =========================================================
# 6) TUM FOLD'LARI CALISTIR
# =========================================================

def run_loso_csp_5band_baseline(session_data, verbose=True, csp_components=None):
    """
    Tum LOSO-session fold'larini canonical 5-band CSP + LDA ile calistirir.
    """
    folds = build_loso_folds(session_data)
    bands = get_canonical_bands()
    csp_components_value = resolve_csp_components(csp_components)

    fold_results = []
    all_predictions = []

    if verbose:
        band_text = ", ".join([f"{name}:{low}-{high}" for name, low, high in bands])
        print("\n===== LOSO-SESSION 5-BAND CSP + LDA BASELINE =====")
        print("Bandlar:", band_text)

    for fold_index, fold in enumerate(folds, start=1):
        if verbose:
            print(f"\nFold {fold_index}")
            print("Train session'lar:", fold["train_sessions"])
            print("Test session:", fold["test_session"])

        X_train, y_train, X_test, y_test = build_train_test_data(session_data, fold)

        if verbose:
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)

        auc, bal_acc, cm, y_pred, y_score = run_one_fold(
            X_train, y_train, X_test, y_test, csp_components=csp_components_value
        )

        if verbose:
            print("ROC-AUC:", round(auc, 4))
            print("Balanced Accuracy:", round(bal_acc, 4))
            print("Confusion Matrix:")
            print(cm)

        result_row = {
            "fold_id": fold_index,
            "test_session": fold["test_session"],
            "train_sessions": ",".join(fold["train_sessions"]),
            "n_train_samples": len(y_train),
            "n_test_samples": len(y_test),
            "n_bands": 5,
            "csp_components_per_band": csp_components_value,
            "roc_auc": round(float(auc), 6),
            "balanced_accuracy": round(float(bal_acc), 6),
            "cm_00": int(cm[0, 0]),
            "cm_01": int(cm[0, 1]),
            "cm_10": int(cm[1, 0]),
            "cm_11": int(cm[1, 1]),
        }

        fold_results.append(result_row)

        for i in range(len(y_test)):
            pred_row = {
                "fold_id": fold_index,
                "test_session": fold["test_session"],
                "true_label": int(y_test[i]),
                "pred_label": int(y_pred[i]),
                "pred_score": float(y_score[i]),
            }
            all_predictions.append(pred_row)

    return fold_results, all_predictions


# =========================================================
# 7) ORTALAMA SONUCLAR
# =========================================================

def get_final_summary(fold_results):
    """
    Fold sonuclarinin ortalamasini sozluk olarak dondurur.
    """
    auc_values = [row["roc_auc"] for row in fold_results]
    bal_values = [row["balanced_accuracy"] for row in fold_results]

    mean_auc = float(np.mean(auc_values))
    mean_bal = float(np.mean(bal_values))

    summary = {
        "n_folds": len(fold_results),
        "mean_roc_auc": mean_auc,
        "mean_balanced_accuracy": mean_bal
    }

    return summary


def print_final_summary(fold_results):
    """
    Fold sonuclarinin ortalamasini ekrana basar.
    """
    summary = get_final_summary(fold_results)

    print("\n===== GENEL SONUC =====")
    print("Fold sayisi:", summary["n_folds"])
    print("Ortalama ROC-AUC:", round(summary["mean_roc_auc"], 4))
    print("Ortalama Balanced Accuracy:", round(summary["mean_balanced_accuracy"], 4))
