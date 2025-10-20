from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def traduccion_nlp(texto_en: str):
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenizar y traducir
    inputs = tokenizer.encode(texto_en, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)

    # Decodificar texto traducido
    texto_es = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return texto_es  # 👈 devuelve solo el string
