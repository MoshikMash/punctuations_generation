import pandas as pd
import streamlit as st
from newspaper import Article
import string
import re

from punctuation_app_api import PunctuationAPI

# st.markdown('<style> textarea {direction: rtl;} </style>', unsafe_allow_html=True)
st.markdown('<style> [textarea] {direction: rtl; text-align: right;} </style>', unsafe_allow_html=True)
st.markdown(
    """<style>table td, table td * {
        vertical-align: top;
    }<style>""", unsafe_allow_html=True)
st.markdown(
    """<style>table {
      table-layout: fixed ;
      width: 100% ;
    }</style>""", unsafe_allow_html=True)
# st.markdown("""<style>button {float: right;}</style>""", unsafe_allow_html=True)


def add_break_lines(original_text, processed_text):
    original_text_break_line_split = original_text.split('\n')
    processed_text_word_split = processed_text.split()
    last_idx = 6
    indices_to_break_line = []
    for paragraph in original_text_break_line_split:
        if not paragraph:
            continue
        paragraph_words = paragraph.split()
        for i in range(last_idx, len(processed_text_word_split)):
            count = 0
            for j in range(6):
                if processed_text_word_split[i - j] == paragraph_words[-1 - j]:
                    count += 1
            if count >= 3:
                indices_to_break_line.append(i)
                last_idx = i

    processed_text = processed_text.split()
    for idx in reversed(indices_to_break_line):
        if '\"' in processed_text[idx + 1] or ')' in processed_text[idx + 1]:
            processed_text.insert(idx + 2, '\n\n')
        else:
            processed_text.insert(idx + 1, '\n\n')

    return ' '.join(processed_text)


def remove_extra_newlines(s):
    # Replace all occurrences of three or more consecutive newlines with just two newlines
    while '\n\n\n' in s:
        s = s.replace('\n\n\n', '\n\n')

    return s


def process_text(user_input):
    punctuation_processor = PunctuationAPI(chunk_size=128)
    if user_input.startswith('http') or user_input.startswith('www'):
        article = Article(user_input, language='he')
        article.download()
        article.parse()
        original_text = article.text
        punctuation_processor.evaluate(original_text)
    else:
        punctuation_processor.evaluate(user_input)
        original_text = user_input

    output_text = punctuation_processor.generated_text
    return original_text, output_text


def main():
    st.title("Punctuation Processor")

    text = """Enter some Hebrew text (with or without punctuations) or insert a
     link to an Hebrew article (at least 128 words)"""
    user_input = st.text_area(text)

    if st.button("Process"):
        original_text, processed_text = process_text(user_input)
        original_text = original_text.replace('\n', '')

        # processed_text = add_break_lines(original_text, processed_text)
        original_text = remove_extra_newlines(original_text)
        original_text = original_text.replace('.', '<mark>.</mark>')
        original_text = original_text.replace(',', '<mark>,</mark>')
        original_text = original_text.replace(':', '<mark>:</mark>')
        original_text = original_text.replace('?', '<mark>:</mark>')
        processed_text = processed_text.replace('.', '<mark>.</mark>')
        processed_text = processed_text.replace(',', '<mark>,</mark>')
        processed_text = processed_text.replace(':', '<mark>:</mark>')
        processed_text = processed_text.replace('?', '<mark>:</mark>')

        processed_df = pd.DataFrame({'טקסט מקור': [original_text], 'טקסט מתוקן': [processed_text]})

        processed_df = processed_df.applymap(lambda x: x.replace('\n', '<br>'))

        table_html = processed_df.to_html(index=False, escape=False)

        table_html = table_html.replace('<td>', '<td><div class="cell" id="{{col}}-{{row}}">')
        table_html = table_html.replace('</td>', '</div></td>')

        table_html = table_html.replace('<table', '<table style="direction: rtl;"')

        # Display the HTML table in Streamlit
        st.write(table_html, unsafe_allow_html=True)

        st.write()


if __name__ == "__main__":
    main()
