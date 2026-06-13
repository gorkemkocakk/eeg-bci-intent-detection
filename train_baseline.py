# train_baseline.py

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


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
            "Cross-session baseline için en az 2 session feature dosyası gerekir."
        )

    folds = []

    # Session sirasi sayisal olarak korunur; burada random shuffle yapilmaz.
    # LOSO-session, modelin baska oturuma genellenmesini olcmek icin her seferinde tek session'i testte birakir.
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
# Burada session olarak ayırdığımız için train test yapacağız.
# 0.3 0.7 gibi oranlarla bölmek yerine, session bazlı LOSO yapacağız.
# Yani her fold'da bir session'ı test olarak ayıracağız, kalan session'ları train olarak kullanacağız.
# Böylece modelin farklı session'larda ne kadar genelleme yapabildiğini göreceğiz.

def build_train_test_data(session_data, fold):
    """
    Verilen fold için train ve test matrislerini üretir.
    """
    X_train_list = []
    y_train_list = []

    # Egitim verisi yalnizca test session disindaki session'lardan birlestirilir.
    # Test session'in feature dagilimi train tarafindaki hicbir fit islemine dahil edilmez.
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
# 4) TEK FOLD ÇALIŞTIR
# =========================================================

def run_one_fold(X_train, y_train, X_test, y_test):
    """
    Tek bir LOSO fold'unu çalıştırır.

    Kural:
    - scaler sadece train'de fit edilir
    - test sadece transform edilir
    - sonra LDA eğitilir
    """
    # StandardScaler train'de fit edilir; test verisi sadece ayni ortalama/std ile transform edilir.
    # Bu ayrim korunmazsa test session bilgisi egitime sizabilir ve skor yapay yukselebilir.
    # Train-only scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # İlk classifier
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train_scaled, y_train)

    y_pred = lda_model.predict(X_test_scaled)

    # ROC-AUC için skor
    if hasattr(lda_model, "predict_proba"):
        y_score = lda_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = lda_model.decision_function(X_test_scaled)

    # ROC-AUC sirali skor kalitesini, balanced accuracy ise iki sinifin dengeli basarisini ozetler.
    # 0=ITI/non-control ve 1=feedback/control siniflari dengesiz olabilecegi icin ikisi birlikte raporlanir.
    auc = roc_auc_score(y_test, y_score)
    bal_acc = balanced_accuracy_score(y_test, y_pred)  # Kullanma sebebimiz dengesiz veri durumunda accuracy yanıltıcı olabilir, balanced accuracy ise her sınıfın doğruluğunu ayrı ayrı hesaplayıp ortalamasını alır, böylece dengesiz veri durumlarında daha adil bir değerlendirme sağlar.
    cm = confusion_matrix(y_test, y_pred)  # FP FN TN TP değil, sklearn'de düzen [[TN, FP], [FN, TP]] şeklindedir. Yani cm[0, 0] TN, cm[0, 1] FP, cm[1, 0] FN, cm[1, 1] TP olur.

    return auc, bal_acc, cm, y_pred, y_score


# =========================================================
# 5) TÜM FOLD'LARI ÇALIŞTIR
# =========================================================

def run_loso_baseline(session_data, verbose=True):
    """
    Tüm LOSO-session fold'larını çalıştırır.
    """
    folds = build_loso_folds(session_data)

    fold_results = []
    all_predictions = []

    if verbose:
        print("\n===== LOSO-SESSION BASELINE =====")

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
            "roc_auc": round(float(auc), 6),
            "balanced_accuracy": round(float(bal_acc), 6),
            "cm_00": int(cm[0, 0]),
            "cm_01": int(cm[0, 1]),
            "cm_10": int(cm[1, 0]),
            "cm_11": int(cm[1, 1]),
        }

        fold_results.append(result_row)

        # Tahminleri de session bazlı saklayalım
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
# 6) ORTALAMA SONUÇLAR
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