# -*- coding: utf-8 -*-

# 1. コーパス中最も出現頻度の高い自立語単語を上位n個抽出する

import argparse
import treetaggerwrapper as ttw

# 出力ファイル
OUT_PUT_FILE = 'tmpFiles/most_frequently_words.csv'

# tree-taggerを使って英文を形態素解析する
tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR='/home/kaneko-takayuki/tree-tagger')

# 「名詞/動詞/形容詞/副詞」系統のタグ(参考: http://computer-technology.hateblo.jp/entry/20150824/p1)
pos_tags = ['NN', 'NNS', 'VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ', 'JJ', 'JJR', 'JJS', 'PB', 'PBR', 'PBS']


def extract_self_sufficient_word(sentence):
    # 文章をパース
    parse_results = list(map(lambda x: x.split('\t'), tagger.TagText(sentence)))
    # タグでフィルタリング
    filtered_by_tags_parse_results = filter(lambda result: (len(result) == 3) and (result[1] in pos_tags), parse_results)
    # 原型のみ抽出
    original_words = list(map(lambda result: result[2], filtered_by_tags_parse_results))
    return original_words


def main(corpus, n):
    # 単語カウント用の辞書
    word_count_dict = {}

    # 単語のカウントを行う
    with open(corpus, 'r') as f:
        for line in f:
            original_words = extract_self_sufficient_word(line)
            for word in original_words:
                # 辞書に単語が登録されていなければ、初期値0を入れる
                if word not in word_count_dict:
                    word_count_dict[word] = 0
                # インクリメント
                word_count_dict[word] += 1

    # 出現頻度の高いn単語を出力する
    # 降順ソート
    sorted_count_list = sorted(word_count_dict.items(), key=lambda x: -x[1])
    with open(OUT_PUT_FILE, 'w') as f:
        # 上位n個出力
        for word, count in sorted_count_list[:n]:
            f.write(word + ',' + str(count) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract most frequently n-words')
    parser.add_argument('--corpus', '-c', type=str, help='source domain corpus(english only now)')
    parser.add_argument('--topN', '-n', type=int, default=10000, help='extract top n-words')
    args = parser.parse_args()

    main(args.corpus, args.topN)
