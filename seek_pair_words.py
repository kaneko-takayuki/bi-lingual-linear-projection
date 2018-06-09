# -*- coding: utf-8 -*-

import argparse
from google.cloud import translate

CREDENTIAL_FILE = '/home/kaneko-takayuki/.config/gcloud/application_default_credentials.json'
OUT_PUT_FILE = 'tmpFiles/word_pair.csv'

# Translate Client
translate_client = translate.Client()
target = 'ja'


def translate(text):
    translation = translate_client.translate(
        text,
        target_language=target
    )

    return translation['translatedText']


def main(corpus):
    with open(corpus, 'r') as corpusFile, open(OUT_PUT_FILE, 'w') as outputFile:
        for line in corpusFile:
            en_word, count = line.split(',')
            ja_word = translate(en_word)
            outputFile.write(en_word + ',' + ja_word + ',' + count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seek pair words')
    parser.add_argument('--corpus', '-c', type=str, help='most_frequently_file(csv)')
    args = parser.parse_args()

    main(args.corpus)
