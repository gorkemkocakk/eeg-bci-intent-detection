# train_csp_5band_fs_baseline.py

from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from train_csp_5band_baseline import (
    build_5band_csp_features,
    build_loso_folds,
    build_train_test_data,
    resolve_csp_components,
    sort_session_ids,
)


# =========================================================
# 1) BASIT AYAR
# =========================================================

# 5-band CSP sonucu toplam 20 feature geliyor (5 band * 4 bilesen).
# Baslangic icin sade bir secim: en iyi 10 feature.
K_BEST_DEFAULT = 10


# =========================================================
# 2) K-BEST YARDIMCI FONKSIYON
# =========================================================

def get_k_best_value(n_features, k_best=None):
    """
    Guvenli k degerini belirler.
    """
    if n_features <= 0:
        raise ValueError("Feature sayisi sifir veya negatif olamaz.")

    # None -> mevcut varsayilan davranis (k=10 veya feature sayisi kadar)
    if k_best is None:
        return min(K_BEST_DEFAULT, n_features), False

    # all -> feature selection bypass
    if isinstance(k_best, str) and k_best.lower() == "all":
        # "all" secenegi ablation icindir; feature secimini kapatir ama feature sirasi korunur.
        return n_features, True

    k_value = int(k_best)
    if k_value <= 0:
        raise ValueError("k_best pozitif bir tamsayi veya 'all' olmali.")

    return min(k_value, n_features), False


# =========================================================
# 3) TEK FOLD CALISTIR
# =========================================================

def run_one_fold(X_train, y_train, X_test, y_test, csp_components=None, k_best=None):
    """
    Tek bir LOSO fold'unu calistirir.

    Kural:
    - 5-band CSP sadece train'de fit edilir
    - feature selection sadece train'de fit edilir
    - scaler sadece train'de fit edilir
    - test tarafi sadece transform edilir
    """
    csp_components_value = resolve_csp_components(csp_components)

    n_channels = int(X_train.shape[1])
    if csp_components_value > n_channels:
        raise ValueError(
            f"csp_components ({csp_components_value}) kanal sayisindan buyuk olamaz "
            f"(n_channels={n_channels})."
        )

    X_train_feat, X_test_feat = build_5band_csp_features(
        X_train,
        y_train,
        X_test,
        sfreq=float(config.TARGET_SFREQ),
        csp_components=csp_components_value
    )

    k_value, bypass_fs = get_k_best_value(X_train_feat.shape[1], k_best=k_best)

    # Feature selection supervised bir adimdir; mutual information y_train ile hesaplanir.
    # Selector test feature'larina fit edilirse test session bilgisi modele sizmis olur.
    if bypass_fs:
        X_train_sel = X_train_feat
        X_test_sel = X_test_feat
    else:
        score_func = partial(mutual_info_classif, random_state=int(config.RANDOM_SEED))
        selector = SelectKBest(score_func=score_func, k=k_value)
        X_train_sel = selector.fit_transform(X_train_feat, y_train)
        X_test_sel = selector.transform(X_test_feat)

    # CSP ve feature selection sonrasi olcek farklari LDA'yi etkileyebilir.
    # Scaler da yalnizca train secilmis feature'larinda fit edilir.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

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

    return auc, bal_acc, cm, y_pred, y_score, k_value


# =========================================================
# 4) TUM FOLD'LARI CALISTIR
# =========================================================

def run_loso_csp_5band_fs_baseline(session_data, verbose=True, csp_components=None, k_best=None):
    """
    Tum LOSO-session fold'larini 5-band CSP + FS + LDA ile calistirir.
    """
    folds = build_loso_folds(session_data)
    csp_components_value = resolve_csp_components(csp_components)

    fold_results = []
    all_predictions = []

    if verbose:
        print("\n===== LOSO-SESSION 5-BAND CSP + FS + LDA BASELINE =====")

    for fold_index, fold in enumerate(folds, start=1):
        if verbose:
            print(f"\nFold {fold_index}")
            print("Train session'lar:", fold["train_sessions"])
            print("Test session:", fold["test_session"])

        X_train, y_train, X_test, y_test = build_train_test_data(session_data, fold)

        if verbose:
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)

        auc, bal_acc, cm, y_pred, y_score, k_value = run_one_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            csp_components=csp_components_value,
            k_best=k_best
        )

        if verbose:
            print("Secilen feature sayisi (k):", k_value)
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
            "k_best": int(k_value),
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
# 5) ORTALAMA SONUCLAR
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
