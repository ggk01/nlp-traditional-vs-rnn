# 🧠 NLP - Σύγκριση Παραδοσιακών Μοντέλων και RNNs (Assignment 2)

Αυτό το repository περιλαμβάνει την δεύτερη εργασία του μαθήματος **Φυσική Γλώσσα και Επεξεργασία Κειμένου (NLP)** και εξετάζει τη σύγκριση ανάμεσα σε:

- Παραδοσιακά μοντέλα ταξινόμησης (Naive Bayes, SVM)
- Νευρωνικά δίκτυα (RNNs, LSTMs)
- Pre-trained embeddings (GloVe)
- Διαφορετικά datasets (AG News, IMDB)

Κάθε ενότητα συνοδεύεται από **Jupyter Notebook (.ipynb)** και **έκδοση σε PDF (.pdf)** για παρουσίαση.

---

## 📚 Περιεχόμενα



### Ενότητα Α: Word Embeddings & Αναλύσεις
- `Assignment_2_A.ipynb` – Ανάλυση GloVe και Word2Vec.
- `Assignment_2_A.pdf` – Έκδοση σε PDF.

### Ενότητα B: Παραδοσιακά Μοντέλα
- `Assignment_2_B.ipynb` – Ταξινόμηση AG News με Naive Bayes και SVM.
- `Assignment_2_B.pdf` – Έκδοση σε PDF.

### Ενότητα C: RNN & LSTM σε AG News & IMDB
- `Assignment_2_C.ipynb` – Πλήρης πειραματική διαδικασία:
  - Εκπαίδευση RNNs (1-layer, Bi-directional, 2-layer)
  - Με/Χωρίς GloVe (frozen/unfrozen)
  - AG News + IMDB
- `Assignment2-RNNs.pdf` – Έκδοση σε PDF.

---

## ⚙️ Οδηγίες Εκτέλεσης

Η εργασία μπορεί να αναπαραχθεί τοπικά με χρήση **Conda**.

### 1️⃣ Δημιουργία περιβάλλοντος
```bash
conda env create -f environment.yml
conda activate nlp-assignment-2
