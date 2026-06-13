# load_stieger.py

import json #JSON formatında envanter kaydetmek için
import os
import re # Session ve run isimlerini sayısal sıraya göre sıralamak için regex kullanacağız *GPT 5.4 tarafından önerildi*

from moabb.datasets import Stieger2021

import config #Kendi yazdığımız ayarları Config dosyasından alacağız, böylece kodun geri kalanında config.SUBJECT_ID gibi ifadelerle ayarlara erişebiliriz. Düzenlemeyi kolaylaştırır ve merkezi bir yerden yönetmemizi sağlar.


def sort_names(name_list):
    """
    Session ve run isimlerini sayısal sıraya göre sıralar.
    Örn: session_2, session_10 gibi durumlarda doğru sırayı korur.
    
    Neden gerekli: String sıralamada "session_10" < "session_2" olur,
    bu fonksiyon sayısal değerlere göre sıralar.
    """
    def get_number(name):
        # Son sayıyı bul (session_01_run_2 → 2'yi al)
        numbers = re.findall(r"\d+", str(name))
        if len(numbers) > 0:
            return int(numbers[-1])
        return 9999  # Sayı yoksa listeye sondan ekle

    return sorted(name_list, key=get_number)


def load_stieger_subject(subject_id): #Bu fonksiyon, Stieger2021 dataset'inden tek bir subject'in tüm session ve run verisini yükler.
    """
    Tek bir subject'in tüm mevcut session verisini yükler.

    Dönen yapı:
    subject_data[session_name][run_name] = raw
    """
    dataset = Stieger2021()

    all_data = dataset.get_data(subjects=[subject_id])

    if subject_id not in all_data:
        raise ValueError(f"Subject {subject_id} verisi bulunamadı.")

    subject_data = all_data[subject_id]
    return subject_data

def build_subject_inventory(subject_data):
    """
    Yüklenen subject verisinden session özeti çıkarır.
    Bu fonksiyon trial parse etmez.
    Sadece hangi session var, kaç run var, sfreq kaç, kanal sayısı kaç gibi
    temel bilgileri toplar.
    
    Cross-session genellemesi için: Her session'ın yapısını öğrenip,
    train/test split stratejisini belirlemede yardımcı olur.
    """
    inventory = [] #Session envanteri, her session için temel bilgileri içeren bir liste. Her session için bir sözlük olacak ve bu sözlükte session adı, run sayısı, sampling rate, kanal sayısı gibi bilgiler yer alacak.

    session_names = sort_names(list(subject_data.keys()))

    for session_name in session_names:#Subject'in her bir session'ı için döner. Session'lar genellikle farklı günlerde yapılan kayıtları temsil eder ve her session içinde birden fazla run olabilir.
        runs = subject_data[session_name]#Session içindeki run'ları alır. Run'lar genellikle aynı session içinde yapılan farklı kayıt bloklarını temsil eder. Örneğin, bir session içinde 3 run olabilir ve her run farklı bir süre boyunca kayıt yapılmış olabilir.
        run_names = sort_names(list(runs.keys()))

        first_run = runs[run_names[0]]#

        sfreq = first_run.info["sfreq"]#Sampling rate, yani saniyede kaç örnek alındığı bilgisini verir. Bu bilgi, EEG verisinin zaman çözünürlüğünü anlamak için önemlidir. Genellikle 250 Hz, 500 Hz gibi değerler olabilir.
        ch_names = first_run.ch_names

        total_annotations = 0 #Session içindeki tüm run'ların annotation sayısını toplar. Annotation'lar, EEG verisi üzerinde belirli olayların veya durumların işaretlendiği noktalardır. Örneğin, bir deneme başlangıcı, bir uyarana tepki gibi olaylar annotation olarak kaydedilir. Toplam annotation sayısı, session'ın ne kadar zengin bir veri içerdiğini gösterir.
        total_duration_sec = 0.0

        for run_name in run_names:
            raw = runs[run_name]
            total_annotations += len(raw.annotations)
            total_duration_sec += raw.n_times / raw.info["sfreq"]

        session_info = {
            "session_name": str(session_name),
            "run_names": [str(x) for x in run_names],
            "n_runs": len(run_names),
            "sampling_rate": sfreq,
            "n_channels": len(ch_names),
            "n_annotations_total": total_annotations,
            "duration_sec_total": round(total_duration_sec, 2),
        }

        inventory.append(session_info)

    return inventory


def print_inventory(inventory):
    """
    Session özetini ekrana yazdırır.
    """
    print("\n===== SUBJECT SESSION ENVANTERİ =====\n")

    for session in inventory:
        print("Session:", session["session_name"])
        print("  Run sayısı:", session["n_runs"])
        print("  Runlar:", session["run_names"])
        print("  Sampling rate:", session["sampling_rate"])
        print("  Kanal sayısı:", session["n_channels"])
        print("  Toplam annotation:", session["n_annotations_total"])
        print("  Toplam süre (sn):", session["duration_sec_total"])
        print("-" * 40)


def save_inventory(inventory, subject_id):
    """
    Session envanterini json olarak kaydeder.
    """
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    save_path = os.path.join(
        config.OUTPUT_DIR,
        f"subject_{subject_id}_inventory.json"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=4, ensure_ascii=False)

    print(f"\nEnvanter kaydedildi: {save_path}")


def main(): #Programın ana fonksiyonu, burada tüm işlemler sırayla yapılır. Önce config dosyasından subject_id alınır, sonra bu subject'in verisi yüklenir, envanteri oluşturulur, ekrana yazdırılır ve json olarak kaydedilir.
    subject_id = config.SUBJECT_ID

    print(f"Subject {subject_id} yükleniyor...")

    subject_data = load_stieger_subject(subject_id)

    inventory = build_subject_inventory(subject_data)

    print_inventory(inventory)
    save_inventory(inventory, subject_id)

    print("\nVeri yükleme tamamlandı.")


if __name__ == "__main__":#Bu blok, Python dosyası doğrudan çalıştırıldığında main() fonksiyonunu çağırır. Eğer bu dosya başka bir yerden import edilirse, main() fonksiyonu otomatik olarak çalışmaz. Bu sayede, bu dosyayı modül olarak kullanmak isteyenler main() fonksiyonunu manuel olarak çağırabilirler.
    main()