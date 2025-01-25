
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return tokenizer, summarizer, translator

tokenizer, summarizer, translator = load_models()

import streamlit as st
LANGUAGE_CODES = {
    "en": "en_XX",  # English
    "fr": "fr_XX",  # French
    "de": "de_DE",  # German
    "ru": "ru_RU",  # Russian
    "hi": "hi_IN",  # Hindi
    "mr": "mr_IN",  # Marathi
    "ja": "ja_XX",  # Japanese
   
}

def detect_language(text):
    lang_code = detect(text)
    return lang_code


def summarize_text(text, lang_code):
    mbart_lang_code = LANGUAGE_CODES.get(lang_code, "en_XX")  # Default to English if unsupported
    inputs = tokenizer(
        f"<{mbart_lang_code}>{text}", 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    )
    summary_ids = summarizer.model.generate(
        inputs["input_ids"], 
        max_length=100, 
        min_length=30, 
        length_penalty=2.0, 
        num_beams=4
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def translate_to_english(text, lang_code):
    mbart_lang_code = LANGUAGE_CODES.get(lang_code, "en_XX")  # Default to English if unsupported
    inputs = tokenizer(
        f"<{mbart_lang_code}>{text}", 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    )
    translated_ids = translator.model.generate(
        inputs["input_ids"], 
        max_length=100, 
        length_penalty=2.0, 
        num_beams=4
    )
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text


st.title("Multilingual Summarization and Translation App")
st.markdown("""This app detects the language of the input text, summarizes it in the same language, and translates it into English.""")


user_input = st.text_area("Enter text in any language:", "")

if st.button("Process Text"):
    if user_input.strip():
        
        lang_code = detect_language(user_input)
        st.write(f"**Detected Language Code:** {lang_code}")
        
        if lang_code not in LANGUAGE_CODES:
            st.warning(f"The detected language ({lang_code}) is not supported by the model.")
        else:
            try:
               
                summary = summarize_text(user_input, lang_code)
                st.write(f"### Summarized Text ({lang_code}):")
                st.write(summary)

                
                translation = translate_to_english(summary, lang_code)
                st.write("### Translated Text (English):")
                st.write(translation)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please enter some text to process.")
