# train_csp_baseline.py

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config


# =========================================================
# 1) KÜÇÜK YARDIMCI FONKSİYON
# =========================================================

def sort_session_ids(session_ids):
    """
    Session isimlerini sayısal sıraya göre sıralar.
    """
    return sorted(session_ids, key=lambda x: int(x))


# =========================================================
# 2) LOSO-SESSION FOLD'LARI
# =========================================================

def build_loso_folds(session_data):
    """
    Leave-one-session-out fold listesini üretir.

    Her fold:
    - 1 session test
    - kalan session'lar train
    """
    session_ids = sort_session_ids(list(session_data.keys()))

    if len(session_ids) < 2:
        raise ValueError(
            "Cross-session CSP baseline için en az 2 session window dosyası gerekir."
        )

    folds = []

    # Session'lar karistirilmadan sirayla test fold'u yapilir.
    # Bu, pseudo-online senaryoda ayni subject icinde cross-session genellemesini olcer.
    for test_session in session_ids:
        train_sessions = [s for s in session_ids if s != test_session]

        fold = {
            "test_session": test_session,
            "train_sessions": train_sessions
        }

        folds.append(fold)

    return folds


# =========================================================
# 3) TRAIN / TEST MATRİSLERİNİ OLUŞTUR
# =========================================================

def build_train_test_data(session_data, fold):
    """
    Verilen fold için train ve test matrislerini üretir.
    """
    X_train_list = []
    y_train_list = []

    # CSP ham pencere sinyaliyle calisir; burada test session train listesine hic eklenmez.
    # Boylece spatial pattern hesabinda test oturumunun kovaryans bilgisi kullanilmaz.
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
# 4) CSP FEATURE ÇIKARIMI (TRAIN-ONLY FIT)
# =========================================================

def extract_log_variance(csp_space_data):
    """
    CSP uzayındaki zaman serilerinden log-variance feature üretir.

    Girdi:
    csp_space_data -> (n_epochs, n_components, n_times)

    Çıktı:
    features -> (n_epochs, n_components)
    """
    variance = np.var(csp_space_data, axis=2)
    log_variance = np.log(variance + 1e-10)
    return log_variance


def fit_transform_csp_features(X_train, y_train, X_test):
    """
    CSP'yi sadece train veride fit eder.
    Sonra train ve test için aynı CSP ile transform uygular.
    """
    # CSP sinif etiketlerini kullanarak spatial filtre ogrenir.
    # Bu nedenle fit_transform yalnizca training veride yapilmali, test sadece transform edilmelidir.
    csp_model = CSP(
        n_components=int(config.CSP_COMPONENTS),  # Başlangıç için 4
        reg="ledoit_wolf",  # Sayısal kararlılık için hafif düzenleme
        log=None,  # log-variance'ı aşağıda açıkça biz alıyoruz
        transform_into="csp_space",
        norm_trace=False
    )

    X_train_csp = csp_model.fit_transform(X_train, y_train)
    X_test_csp = csp_model.transform(X_test)

    X_train_features = extract_log_variance(X_train_csp)
    X_test_features = extract_log_variance(X_test_csp)

    return X_train_features, X_test_features


# =========================================================
# 5) TEK FOLD ÇALIŞTIR
# =========================================================

def run_one_fold(X_train, y_train, X_test, y_test):
    """
    Tek bir LOSO fold'unu çalıştırır.

    Kural:
    - CSP sadece train'de fit edilir
    - test sadece train-CSP ile transform edilir
    - scaler sadece train'de fit edilir
    - sonra LDA eğitilir
    """
    X_train_feat, X_test_feat = fit_transform_csp_features(X_train, y_train, X_test)

    # CSP sonrasi feature olcegi fold'a gore degisebilir.
    # Scaler da CSP gibi sadece train'de fit edilerek test session etkisi disarida tutulur.
    # Train-only scaler
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
# 6) TÜM FOLD'LARI ÇALIŞTIR
# =========================================================

def run_loso_csp_baseline(session_data, verbose=True):
    """
    Tüm LOSO-session fold'larını CSP + LDA ile çalıştırır.
    """
    folds = build_loso_folds(session_data)

    fold_results = []
    all_predictions = []

    if verbose:
        print("\n===== LOSO-SESSION CSP + LDA BASELINE =====")

    for fold_index, fold in enumerate(folds, start=1):
        if verbose:
            print(f"\nFold {fold_index}")
            print("Train session'lar:", fold["train_sessions"])
            print("Test session:", fold["test_session"])

        X_train, y_train, X_test, y_test = build_train_test_data(session_data, fold)

        if verbose:
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)

        auc, bal_acc, cm, y_pred, y_score = run_one_fold(X_train, y_train, X_test, y_test)

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
            "csp_components": int(config.CSP_COMPONENTS),
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
# 7) ORTALAMA SONUÇLAR
# =========================================================

def get_final_summary(fold_results):
    """
    Fold sonuçlarının ortalamasını sözlük olarak döndürür.
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
    Fold sonuçlarının ortalamasını ekrana basar.
    """
    summary = get_final_summary(fold_results)

    print("\n===== GENEL SONUÇ =====")
    print("Fold sayısı:", summary["n_folds"])
    print("Ortalama ROC-AUC:", round(summary["mean_roc_auc"], 4))
    print("Ortalama Balanced Accuracy:", round(summary["mean_balanced_accuracy"], 4))
