# config.py

import os


# =========================================================
# 1) PROJEDE EN ÇOK DEĞİŞTİRECEĞİMİZ AYARLAR
# =========================================================

# Şu an tek subject ile çalışacağız.
# Hangi subject'i indirdiysen burada onu yazarsın.
SUBJECT_ID = 1

# Random işlemler için sabit seed kullanmak iyi olur, böylece sonuçlar tekrarlanabilir olur.
RANDOM_SEED = 42


# =========================================================
# 2) PROJENİN SABİT MANTIĞI
# =========================================================

DATASET_NAME = "Stieger2021"
PROBLEM_TYPE = "control_vs_non_control yani 1/0 start stop ayrımı"
EVALUATION_TYPE = "pseudo_online"

# Etiket mantığı
LABEL_ITI = 0 #ITI: Feedback başlamadan önceki bekleme süresi, yani kontrolsüz durum. Bu süre boyunca beyin aktivitesi normal olarak kabul edilir. Açılımı: ITI, Inter-Trial Interval'ın kısaltmasıdır.
LABEL_FEEDBACK = 1 #Burada Label_FEEDBACK, feedback başladığında ortaya çıkan beyin aktivitesini temsil eder. Bu durum, kontrol durumunu gösterir ve modelin bu iki durumu ayırt etmesi beklenir.

# Ana split mantığı
SPLIT_TYPE = "LOSO_session" #LOSO şu demek: Leave-One-Session-Out, yani her seansı sırayla test seti yaparak modeli eğitmek. Bu yöntem, modelin farklı seanslardaki performansını değerlendirmek için kullanılır.

# Session'ları karıştırmak istemiyoruz
SESSION_SHUFFLE = False


# =========================================================
# 3) PREPROCESSING AYARLARI
# =========================================================

# İlk sprintte önerilen başlangıç
TARGET_SFREQ = 250 #Downsample yapmak, işlem süresini azaltır ve genellikle 250 Hz, 30 Hz'ye kadar olan beyin dalgalarını yakalamak için yeterlidir.

# İlk çalışan baseline için dar bant
LOW_FREQ = 8
HIGH_FREQ = 30

# İleride hoca isterse 5 bant analizine geçebilmek için geniş bant da tanımlandı.
BROAD_LOW_FREQ = 0.5
BROAD_HIGH_FREQ = 50

CANONICAL_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


# =========================================================
# 4) WINDOWING AYARLARI
# =========================================================

# Pencereleri feedback başlangıcına göre çıkaracağız
WINDOW_REFERENCE = "feedback_onset" #Window Reference: Pencereleri, feedback'in başladığı anı referans alarak çıkarmak Yani her pencere, feedback başlangıcına göre hizalanacak.

WINDOW_SIZE_SEC = 2.0 #Stieger 2021'de feedback 2 saniye sürdüğü için, bu süre boyunca beyin aktivitesini analiz etmek mantıklı olabilir. Bu nedenle, pencere boyutunu 2 saniye olarak belirlendi.
STRIDE_SEC = 1.0 #Stieger 2021'deki gibi 1 saniyelik adımlarla pencereler oluşturmak, modelin her saniyede bir yeni bilgi almasını sağlar ve böylece daha dinamik bir analiz yapabiliriz. Bu nedenle, stride'ı 1 saniye olarak belirlendi.


# =========================================================
# 5) FEATURE VE MODEL AYARLARI
# =========================================================

# İlk sürümde en basit başlangıç:
# önce bandpower + LDA
FEATURE_TYPE = "bandpower"
CLASSIFIER = "LDA"

# Sonra istersek CSP + LDA'ya geçeriz
CSP_COMPONENTS = 4 #CSP, Common Spatial Patterns'ın kısaltmasıdır. EEG verilerinde sınıflar arasındaki farkları ortaya çıkarmak için kullanılan bir yöntemdir. CSP_COMPONENTS, bu yöntemde kaç bileşen kullanılacağını belirler. Genellikle 4-6 bileşen seçilir, çünkü bu sayede sınıflar arasındaki ayrımı iyi yakalayabiliriz.

# Değerlendirmede ana metrik
PRIMARY_METRIC = "roc_auc" #ROC AUC, Receiver Operating Characteristic - Area Under the Curve'ın kısaltmasıdır. Bu metrik, modelin sınıfları ne kadar iyi ayırdığını ölçer. 0.5 rastgele tahmin anlamına gelirken, 1.0 mükemmel ayrım anlamına gelir.


# =========================================================
# 6) KLASÖR YOLLARI
# =========================================================

# Proje klasör yapısı basit olsun diye burada tutuluyor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")#Data DIR: Ham verilerin saklandığı klasör. Bu klasör, indirilen EEG verilerini içerir. Her subject için ayrı bir alt klasör olabilir, örneğin data/subject_1, data/subject_2 gibi. Bu yapı, verilerin düzenli ve erişilebilir olmasını sağlar.
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw") #Raw Data DIR: Ham EEG verilerinin saklandığı alt klasör. Bu klasör, doğrudan cihazdan alınan ham EEG verilerini içerir. Bu veriler genellikle işlenmemiş ve büyük boyutlu olabilir, bu yüzden ayrı bir klasörde tutulur.
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs") #Output DIR: İşlenmiş verilerin, özelliklerin, modellerin ve sonuçların saklandığı klasör. Bu klasör, projenin çıktılarının düzenli bir şekilde saklanmasını sağlar. İçinde trial_tables, features, models gibi alt klasörler olabilir. Örneğin, outputs/trial_tables/session_1_trial_table.csv gibi dosyalar burada yer alır.
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results") #Results DIR: Model değerlendirme sonuçlarının saklandığı klasör. Bu klasör, model performans metriklerini, grafiklerini ve diğer değerlendirme çıktılarının düzenli bir şekilde saklanmasını sağlar. Örneğin, outputs/results/session_1_results.csv gibi dosyalar burada yer alır.
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")#Log DIR: Programın çalışması sırasında oluşan log dosyalarının saklandığı klasör. Bu klasör, hata mesajları, uyarılar ve diğer önemli bilgilerin kaydedildiği log dosyalarını içerir. Log dosyaları, programın çalışması sırasında ortaya çıkan sorunları tespit etmek ve düzeltmek için kullanılır. Örneğin, outputs/logs/session_1_log.txt gibi dosyalar burada yer alır.


# =========================================================
# 7) KLASÖRLERİ OLUŞTURAN BASİT FONKSİYON
# =========================================================
# Proje klasör yapısını baştan oluşturmak için kullanabileceğimiz bir fonksiyon. Program başında bir kez çağırarak gerekli klasörlerin var olduğundan emin olabiliriz.
def create_folders():
    """
    Gerekli çıktı klasörlerini oluşturur.
    Program başında bir kez çağırabiliriz.
    """
    folders = [OUTPUT_DIR, RESULTS_DIR, LOG_DIR]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


# =========================================================
# 8) KONTROL AMAÇLI KÜÇÜK TEST
# =========================================================
# Bu bölüm, config dosyasının doğru şekilde çalıştığını ve klasörlerin oluşturulduğunu kontrol etmek için kullanılabilir. Programı çalıştırdığınızda bu mesajları görmelisiniz.
if __name__ == "__main__":
    create_folders()

    print("Config dosyası çalıştı.")
    print("Subject ID:", SUBJECT_ID)
    print("Dataset:", DATASET_NAME)
    print("Problem tipi:", PROBLEM_TYPE)
    print("Etiketler -> ITI:", LABEL_ITI, "| Feedback:", LABEL_FEEDBACK)
    print("Pencere:", WINDOW_SIZE_SEC, "sn")
    print("Stride:", STRIDE_SEC, "sn")
    print("Bant:", LOW_FREQ, "-", HIGH_FREQ, "Hz")
    print("Split:", SPLIT_TYPE)