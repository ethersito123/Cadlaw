import os
import re
from PyPDF2 import PdfReader
import sentencepiece as spm
from langdetect import detect

# -----------------------------
# 1️⃣ Funciones para procesar PDFs y texto
# -----------------------------

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF individual."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text):
    """
    Limpia el texto:
    - Normaliza espacios
    - Mantiene letras, números, puntuación y emojis
    """
    text = re.sub(r"\s+", " ", text)  # reemplaza saltos de línea y múltiples espacios
    text = text.strip()
    return text

def detect_language(text):
    """Detecta idioma principal (en, fr, es, etc.)"""
    try:
        return detect(text)
    except:
        return "en"

# -----------------------------
# 2️⃣ Generar corpus genérico
# -----------------------------

def generate_generic_corpus(folder, corpus_file, batch_size=50):
    """
    Extrae texto de PDFs o TXT en lotes y genera corpus.
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".pdf", ".txt"))]
    total_files = len(files)
    print(f"Se encontraron {total_files} archivos para procesar.")

    with open(corpus_file, "w", encoding="utf-8") as f_out:
        for i in range(0, total_files, batch_size):
            batch_files = files[i:i+batch_size]
            batch_texts = []
            for file in batch_files:
                try:
                    if file.endswith(".pdf"):
                        text = extract_text_from_pdf(file)
                    else:
                        with open(file, "r", encoding="utf-8") as f_in:
                            text = f_in.read()
                    text = clean_text(text)
                    batch_texts.append(text)
                except Exception as e:
                    print(f"Error procesando {file}: {e}")
            f_out.write("\n".join(batch_texts) + "\n")
            print(f"Procesados {min(i+batch_size, total_files)}/{total_files} archivos")

# -----------------------------
# 3️⃣ Entrenar tokenizer Unigram genérico
# -----------------------------

def train_generic_tokenizer(corpus_file, model_prefix, vocab_size=32000, character_coverage=1.0):
    """
    Entrena tokenizer Unigram para MoE genérico tipo ChatGPT.
    """
    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=character_coverage,  # 1.0 = todos los caracteres
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<|endoftext|>"]  # símbolo genérico útil para LLM
    )
    print(f"Tokenizer entrenado: {model_prefix}.model y {model_prefix}.vocab")

# -----------------------------
# 4️⃣ Probar tokenizer
# -----------------------------

def test_tokenizer(model_file, sample_text):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    tokens = sp.encode(sample_text, out_type=str)
    ids = sp.encode(sample_text, out_type=int)
    print("Tokens:", tokens)
    print("IDs:", ids)

# -----------------------------
# 5️⃣ Main
# -----------------------------

if __name__ == "__main__":
    folder = "ruta/a/tus/documentos"  # PDFs o TXT
    corpus_file = "generic_corpus.txt"
    model_prefix = "generic_moe_tokenizer"

    # 1. Generar corpus genérico
    generate_generic_corpus(folder, corpus_file, batch_size=50)

    # 2. Entrenar tokenizer Unigram genérico
    train_generic_tokenizer(corpus_file, model_prefix, vocab_size=32000)

    # 3. Probar tokenizer
    sample_text = "Hello! This is a test sentence for a generic MoE model."
    test_tokenizer(f"{model_prefix}.model", sample_text)
