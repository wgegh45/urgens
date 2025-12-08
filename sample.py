#!/usr/bin/env python
# encoding: utf-8

import nltk # NLTK（Natural Language Toolkit）は、Python で自然言語処理（NLP）を行うための代表的なライブラリ
nltk.data.path.append(r".\nltk_data") # NLTK (Natural Language Toolkit) のデータパスを追加
                                     # NLTK が .\nltk_data フォルダから必要なリソースを探す
                                      
'''                                      
nltk.download('wordnet')   # 英語の大規模な語彙データベース - 単語の 同義語 (synonyms)、反義語 (antonyms)、上位語 (hypernyms)、下位語 (hyponyms) などの調査に使用
nltk.download('brown')     # Brown Corpus - 英語の代表的なコーパスのひとつで、1961年にアメリカ英語の文章をジャンル別に収集したもの
                           # - NLTK では、このコーパスを使って 言語研究・統計解析・機械学習の訓練などが可能
nltk.download('stopwords') # 自然言語処理でよく使われる「頻出するが意味的にはあまり重要でない単語」のリスト
nltk.download('punkt')     # 英語の文や単語を分割するための NLTK の事前学習済みトークナイザー 
                           # Punkt Sentence Tokenizer Pickle形式 NLTK 初期から
nltk.download('punkt_tab') # 英語の文や単語を分割するための NLTK の事前学習済みトークナイザー   
                           # Punkt Sentence Tokenizer (Tabular) JSON形式 NLTK 3.8以降（2023年～）                           
nltk.download('averaged_perceptron_tagger')     # 従来から存在する英語用の品詞タグ付けモデル Averaged Perceptron アルゴリズムを使って、単語に品詞ラベル（名詞、動詞、形容詞など）を付与
nltk.download('averaged_perceptron_tagger_eng') # NLTK 3.8（2023年～）以降で追加された英語用の品詞タグ付けモデル
                                                # 実際の中身は従来の averaged_perceptron_tagger とほぼ同じ英語モデル、英語用であることを示すため_engが付く
'''
'''                         
# 必要なNLTKデータを自動ダウンロード（変更点）
try:
    nltk.data.find('tokenizers/punkt') # tokenizers/punktフォルダがある場合
except LookupError:
    nltk.download('punkt') # 英語の文や単語を分割するための NLTK の事前学習済みトークナイザーをダウンロード 
                           # 一度ダウンロードすれば、次回以降は再ダウンロード不要   Punkt Sentence Tokenizer Pickle形式 NLTK 初期から
try:
    nltk.data.find('tokenizers/punkt_tab') # tokenizers/punkt_tabフォルダがある場合 
except LookupError:
    nltk.download('punkt_tab') # 英語の文や単語を分割するための NLTK の事前学習済みトークナイザーをダウンロード   
                               # Punkt Sentence Tokenizer (Tabular) JSON形式 NLTK 3.8以降（2023年～）
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng') # taggers/averaged_perceptron_tagger_engフォルダがある場合
except LookupError:
    nltk.download('averaged_perceptron_tagger')     # 従来から存在する英語用の品詞タグ付けモデル Averaged Perceptron アルゴリズムを使って、単語に品詞ラベル（名詞、動詞、形容詞など）を付与
    nltk.download('averaged_perceptron_tagger_eng') # NLTK 3.8（2023年～）以降で追加された英語用の品詞タグ付けモデル
                                                    # 実際の中身は従来の averaged_perceptron_tagger とほぼ同じ英語モデル、英語用であることを示すため_eng
'''
                                                    
'''
オフライン実行結果
基本単語数 そのまま
年情報の種類 0
年付き単語 0
都市名 0
都市名派生語 0

オンライン実行結果（推定68-150行）
**基本単語:**
- apple, steve jobs, microsoft, tech giant, apple steve, ...

**Wikipedia年情報:**
- 1976（Apple創業年）
- 1955（Steve Jobs生年）
- 1975（Microsoft創業年）

**Wikipedia都市情報:**
- cupertino, sanfrancisco, seattle, redmond, ...

**最終単語リスト例:**
apple
stevejobs
microsoft
techgiant
applesteve
...
apple1976
apple1955
apple1975
stevejobs1976
stevejobs1955
microsoft1975
cupertino
cupertino1976
seattle1975
...

オフライン実行結果（推定30-50行）
**基本単語のみ:**
apple
stevejobs
microsoft
techgiant
applesteve
...

'''

import tweepy  # Twitter API操作用
import csv
from textblob import TextBlob  # 自然言語処理（感情分析、品詞タグ付け等）
import operator
from nltk.tag import pos_tag  # 品詞タグ付け (POS tagging)（NNP=固有名詞の抽出に使用）
from nltk.corpus import stopwords  # 一般的な単語（the, is, at等）の除外用
import re, string
from collections import Counter  # 単語の出現回数をカウント
from textblob.wordnet import Synset  # WordNet（単語の意味的類似度計算用）
import sys
import wikipedia  # Wikipedia APIで人物情報取得（生年等）
import re
import warnings
import exrex  # 正規表現からランダム文字列生成
import argparse
import os
from hurry.filesize import size
from hurry.filesize import alternative
from geotext import GeoText  # テキストから地名抽出
from colorama import Fore, Back, Style  # ターミナル出力の色付け
import requests
from bs4 import BeautifulSoup  # HTML解析

warnings.filterwarnings("ignore", category=UserWarning, module='bs4') 
# 発生元が BeautifulSoup (bs4) モジュールの警告UserWarningを無視するよう指示

# ========================================
# Twitter API認証情報（使用前に設定が必要）
# ========================================
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

print('''     
            Utku Sen's
             _____  _               _ _       _                            
            |  __ \| |             | (_)     | |                            
            | |__) | |__   ___   __| |_  ___ | | __ _ 
            |  _  /| '_ \ / _ \ / _` | |/ _ \| |/ _` |                       
            | | \ \| | | | (_) | (_| | | (_) | | (_| |
            |_|  \_\_| |_|\___/ \__,_|_|\___/|_|\__,_|                           

Personalized wordlist generation with NLP, by analyzing tweets. (A.K.A crunch2049)                        

''')

# ========================================
# カスタムストップワード読み込み
# ========================================
# 除外したい単語リスト（パスワードに不適切な単語等）
with open("stopwords.txt") as f: # stopwords.txtを読み込んで格納する
    content = f.readlines() # ['is\t\n', 'are\t\n', ・・・ \n', 'http, 'https']
extra_stopwords = [x.strip() for x in content] # ['is', 'are', ・・・ , 'http', 'https'] \tや\nを取り除く

# ========================================
# 関数: Twitterからツイート取得
# ========================================
def get_all_tweets(screen_name):
    """
    指定されたTwitterユーザーの全ツイートを取得
    - リツイートは除外
    - URL部分（https://t.co）は削除
    - @メンションで始まるツイートは除外
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # <tweepy.auth.OAuthHandler object at 0x0000018DBFFBCCD0>
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth) # <tweepy.api.API object at 0x0000018DC761A5D0>
    alltweets = []  
    
    # 最初の200ツイートを取得（Twitter APIの上限）
    new_tweets = api.user_timeline(screen_name = screen_name,count=200, include_rts=False,tweet_mode = 'extended')
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    
    # さらに古いツイートを繰り返し取得
    while len(new_tweets) > 0:
      new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, include_rts=False,tweet_mode = 'extended')
      alltweets.extend(new_tweets)
      oldest = alltweets[-1].id - 1
      print("Downloaded %s tweets.." % (len(alltweets)))
    
    # ツイートテキストを結合
    outtweets = ""
    for tweet in alltweets:
        # @メンション（返信）は除外
        if tweet.full_text.encode("utf-8").startswith('@'):
            pass
        else:
            # URL部分を削除して結合
            outtweets += (tweet.full_text.encode("utf-8").split('https://t.co')[0])
    return outtweets

# ========================================
# 関数: 名詞句（noun phrases）の抽出
# ========================================            
'''
def find_noun_phrases(string): # Apple was founded by Steve Jobs. Microsoft is another tech giant.
    """
    テキストから重要な名詞句を抽出
    例: "natural language processing" のような複数単語の名詞
    
    戻り値: 出現頻度順の上位15個の名詞句リスト
    """
    noun_counts = {}
    
#    try:
#        blob = TextBlob(string.decode('utf-8'))
#    except:
#        print("Error occured")
#        return None
#    if blob.detect_language() != "en":
#        print("Tweets are not in English")
#        sys.exit(1)    
    # 変更 言語検出失敗時も処理を続行
    try:
        # bytes型かstr型かを判定
        if isinstance(string, bytes):
            text = string.decode('utf-8', errors='ignore')
        else:
            text = string # こっちに来る　Apple was founded by Steve Jobs. Microsoft is another tech giant.
                              
        blob = TextBlob(text) # テキストを自然言語処理用のオブジェクトに変換して、簡単に操作できるようにする Apple was founded by Steve Jobs. Microsoft is another tech giant.
    except Exception as e:
        print(f"Error in find_noun_phrases: {e}") # 名詞句（noun phrases）のエラー
        return []  # エラー時は空リストを返す
    
    # 英語のツイートのみ対象
    try:
        if blob.detect_language() != "en": # 英語か判定
            print("Tweets are not in English") # ツイートは英語ではありません
            sys.exit(1)
    except:
        print("Warning: Could not detect language, assuming English") 
        # 警告: 言語を検出できませんでした。英語を想定しています
        

    # 修正: テキスト全体から名詞句の出現回数をカウント
    text_lower = text.lower() # 追加

    # デバッグ出力追加
    print(f"DEBUG: blob.noun_phrases = {blob.noun_phrases}")

    # TextBlobで名詞句（noun phrases）を抽出
    for noun in blob.noun_phrases: # apple   steve jobs   microsoft
        # ストップワードや短すぎる単語は除外
        if noun in stopwords.words('english') or noun in extra_stopwords or noun == '' or len(noun) < 3:
            # NLTK が提供する英語のストップワードリストに含まれている、stopwords.txtトに含まれている、空文字、3文字未満のとき
            pass 
        else:   
            # 出現回数をカウント(大文字小文字を区別しない、複数語steve jobsはカウントしない)
            # blob.words.count(noun) は 完全一致 でカウント
            # blob.words は単語を 個別に分割 するため、複数単語の名詞句はカウントできない
            # 変更
#           noun_counts[noun.lower()] = blob.words.count(noun) # {'apple': 1, 'steve jobs': 0, 'microsoft': 1}
            # 修正: テキスト内の出現回数を正しくカウント
            noun_counts[noun.lower()] = text_lower.count(noun.lower())
            
    # 出現頻度順にソートして上位15個を返す
    sorted_noun_counts = sorted(noun_counts.items(), key=operator.itemgetter(1),reverse=True) # [('apple', 1), ('microsoft', 1), ('steve jobs', 0)]
    return sorted_noun_counts[0:15]
'''
def find_noun_phrases(string):
    noun_counts = {}
    
    try:
        # bytes型かstr型かを判定
        if isinstance(string, bytes):
            text = string.decode('utf-8', errors='ignore')
        else:
            text = string # こっちに来る　Apple was founded by Steve Jobs. Microsoft is another tech giant.
        
        blob = TextBlob(text) # テキストを自然言語処理用のオブジェクトに変換して、簡単に操作できるようにする Apple was founded by Steve Jobs. Microsoft is another tech giant.
    except Exception as e:
        print(f"Error in find_noun_phrases: {e}") # 名詞句（noun phrases）のエラー
        return []
    
    try:
        if blob.detect_language() != "en": # 英語か判定
            print("Tweets are not in English") # ツイートは英語ではありません
            sys.exit(1)
    except:
        print("Warning: Could not detect language, assuming English")
        # 警告: 言語を検出できませんでした。英語を想定しています
        
    text_lower = text.lower()
    
    # 固有名詞を取得（除外用）
    import nltk.data
    try:
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)
    except:
        sentences = text.split('.')
    
    proper_nouns_set = set()
    
    for sentence in sentences:
        sentence_clean = re.sub(r'[^\w\s]', ' ', sentence)
        words = sentence_clean.split()
        if words:
            tagged_sent = pos_tag(words)
            for word, pos in tagged_sent:
                is_proper = ((pos in ['NNP', 'NNPS']) or (pos not in ['JJ', 'JJR', 'JJS'] and word and len(word) >= 3 and word[0].isupper() and word.isalpha()))
                if is_proper:
                    proper_nouns_set.add(word.lower())
    
#   print(f"DEBUG: proper_nouns_set = {proper_nouns_set}")
    
    for noun in blob.noun_phrases:
        # 名詞句全体が固有名詞かチェック（単語単位ではなく）
        noun_lower = noun.lower()

        # 所有格を含む名詞句を除外（シングルクォートとカーリークォート）
        if "'" in noun_lower or "\u2019" in noun_lower:
            continue
        
        # 名詞句が固有名詞セットに完全一致するか、または構成単語に固有名詞が含まれるかチェック
        noun_words = noun_lower.split()
        has_proper_noun = (noun_lower in proper_nouns_set or 
                          any(word in proper_nouns_set for word in noun_words))
        
#        print(f"DEBUG: noun='{noun}', noun_words={noun_words}, has_proper_noun={has_proper_noun}")
        
        if (has_proper_noun or 
            noun in stopwords.words('english') or 
            noun in extra_stopwords or 
            noun == '' or 
            len(noun) < 3):
            pass
        else:
            count = text_lower.count(noun_lower)
            noun_counts[noun_lower] = count
    
    sorted_noun_counts = sorted(noun_counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_noun_counts[0:15]

# ========================================
# 関数: 固有名詞（proper nouns）の抽出
# ========================================
'''
def find_proper_nouns(string):
    """
    テキストから固有名詞を抽出
    例: 人名、地名、組織名など
    
    NLTKの品詞タグ 'NNP' (Proper Noun, Singular) を使用
    戻り値: 出現頻度順の上位15個の固有名詞リスト
    """
    try:
        # 追加 bytes型かstr型かを判定
        if isinstance(string, bytes):
            text = string.decode('utf-8', errors='ignore')
        else:
            text = string # こっちに来る　Apple was founded by Steve Jobs. Microsoft is another tech giant. apple is popular.
 
        pattern = re.compile('[\W_]+', re.UNICODE) # re.compile('[\\W_]+')
        
        # 品詞タグ付け  
        tagged_sent = pos_tag(text.split()) # [('Apple', 'NNP'), ('was', 'VBD'), ('founded', 'VBN'), ('by', 'IN'), ('Steve', 'NNP'), ('Jobs.', 'NNP'), ('Microsoft', 'NNP'), ('is', 'VBZ'), ('another', 'DT'), ('tech', 'JJ'), ('giant.', 'NN'), ('apple', 'NN'), ('is', 'VBZ'), ('popular.', 'JJ')]
        # 名詞 (Nouns) NN:単数形名詞（dog, car）  NNS:複数形名詞（dogs, cars）  NNP:固有名詞・単数（Apple, Steve）  NNPS:固有名詞・複数（Americans, Europeans）
        # 形容詞 (Adjectives)  JJ:形容詞（big, good）  JJR:比較級（bigger, better）   JJS:最上級（biggest, best）         
        # 数字  CD:数詞（100, first）      
        # 感嘆詞  UH:感嘆詞（oh, wow）
        # 代名詞 (Pronouns) PRP	人称代名詞（I, you, he）  PRP$:所有代名詞（my, your, his）   WP:疑問代名詞（who, what） WP$:所有の疑問代名詞（whose）
        # 動詞 (Verbs) VB:原形（go, eat）  VBD:過去形（went, ate）  VBG:現在分詞 / 動名詞（going, eating）  VBN:過去分詞（gone, eaten）  VBP:動詞・現在形（I/you/we/they go） VBZ:三単現（he/she/it goes）
        # 副詞 (Adverbs)  RB:副詞（quickly）  RBR:比較級（faster）  RBS:最上級（fastest）        
        # 冠詞・限定詞  DT:限定詞（the, a, this） PDT:前限定詞（all, both） WDT:疑問限定詞（which）
        # 助動詞 (Modals) MD:can, will, must など              
        # 前置詞・接続詞  IN:前置詞 / 従属接続詞（in, on, because）  CC:等位接続詞（and, but, or）        
        # その他   EX:存在構文の “there”  FW:外来語（ラテン語やフランス語など）  LS:リストマーカー（A., B., C.）  POS:所有格（'s）  SYM:記号（$, %, +）  TO:"to"  RP:小辞（give up の up）
        
        # NNP（固有名詞）のみ抽出し、記号を除去
        propernouns = [re.sub(r'\W+', '', word.lower()) for word,pos in tagged_sent if pos == 'NNP']
        # ['apple', 'steve', 'jobs', 'microsoft']

        last_propernouns = []
        for word in propernouns: # apple
            # ストップワードや短すぎる単語は除外
            if word in stopwords.words('english') or word in extra_stopwords or word == '' or len(word) < 3:
                # NLTK が提供する英語のストップワードリストに含まれている、stopwords.txtトに含まれている、空文字、3文字未満のとき
                pass 
            else:
                last_propernouns.append(word) # apple  steve  jobs  microsoft
 
        # 出現回数をカウントして頻度順にソートして上位15個を返す
        propernouns_dict = dict(Counter(last_propernouns)) # {'apple': 1, 'steve': 1, 'jobs': 1, 'microsoft': 1}
        sorted_propernouns_dict = sorted(propernouns_dict.items(), key=operator.itemgetter(1),reverse=True) # [('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1)]
        return sorted_propernouns_dict[0:15]
    except Exception as e: # 追加
        print(f"Error in find_proper_nouns: {e}")
        return [] # 空リストを返す
'''
def find_proper_nouns(string):
    try:
        if isinstance(string, bytes):
            text = string.decode('utf-8', errors='ignore')
        else:
            text = string
        
        import nltk.data
        try:
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = sent_detector.tokenize(text)
        except:
            sentences = text.split('.')
        
        propernouns = []
        
        for sentence in sentences:
            sentence_clean = re.sub(r'[^\w\s]', ' ', sentence)
            words = sentence_clean.split()
            
            if not words:
                continue
            
            tagged_sent = pos_tag(words)
            current_proper = []
            
            for i, (word, pos) in enumerate(tagged_sent):
                # NNP/NNPSまたは大文字始まりで3文字以上の単語を固有名詞候補に
                is_proper = (pos in ['NNP', 'NNPS']) or (word and len(word) >= 3 and word[0].isupper() and word.isalpha())
                
                if is_proper and word and len(word) > 0:
                    current_proper.append(word)
                else:
                    if current_proper:
                        combined = ' '.join(current_proper).lower()
                        propernouns.append(combined)
                        current_proper = []
            
            if current_proper:
                combined = ' '.join(current_proper).lower()
                propernouns.append(combined)
        
        last_propernouns = []
        
        # 形容詞として使われる単語を除外リストに追加
        adjective_exclusions = {'japanese', 'chinese', 'american', 'european', 'asian'}
        for word in propernouns:
            if (word in stopwords.words('english') or 
                word in extra_stopwords or 
                word in adjective_exclusions or  # 追加
                word == '' or 
                len(word) < 3):
                pass
            else:
                last_propernouns.append(word)
        
        propernouns_dict = dict(Counter(last_propernouns))
        sorted_propernouns_dict = sorted(propernouns_dict.items(), 
                                         key=operator.itemgetter(1), 
                                         reverse=True)
        return sorted_propernouns_dict[0:15]
    except Exception as e:
        print(f"Error in find_proper_nouns: {e}")
        return []



# ========================================
# 関数: 単語間の意味的類似度計算
# ========================================
def word_similarity(word1,word2): # apple,apple  apple,microsoft  apple,steve jobs
    """
    WordNetを使って2つの単語の意味的類似度を計算
    
    例: "dog" と "cat" は類似度が高い
        "dog" と "car" は類似度が低い
    
    戻り値: 0.0〜1.0 の類似度スコア
    """
    try:
        # WordNetから名詞の意味情報を取得
        string1 = Synset(word1+'.n.01')
        string2 = Synset(word2+'.n.01')
        return string1.path_similarity(string2)
    except:
        return 0 # 取得できないとき 　apple,apple apple,microsoft  apple,steve jobs  microsoft,apple  microsoft,microsoft  microsoft,steve jobs  steve jobs,apple  steve jobs,microsoft  steve jobs,steve jobs

# ========================================
# 関数: 類似単語のペア生成
# ========================================    
def mass_similarity_compare(wordlist):
    """
    単語リストから意味的に類似した単語ペアを生成
    
    例: ["football", "baseball"] → "footballbaseball"
    
    類似度0.12以上の単語ペアを結合して新しい単語候補を作成
    パスワードでよく使われる「好きなもの2つを組み合わせる」パターンを再現
    """
    # 追加 wordlistがNoneまたは空の場合の処理
    if wordlist is None or len(wordlist) == 0: # 名詞句   wordlist=[('apple', 2), ('microsoft', 1), ('steve jobs', 0)]
        return []                              # 固有名詞 wordlist=[('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1)]
    
    clean_wordlist = []
    out_wordlist = []
    
    # タプルから単語部分だけ抽出
    for word in wordlist: # ('apple', 2)  ('microsoft', 1)  ('steve jobs', 0)    ('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1)
        clean_wordlist.append(word[0]) # ['apple', 'microsoft', 'steve jobs']    ['apple', 'steve', 'jobs', 'microsoft']
    
    # 全単語ペアの類似度を計算
    for word in clean_wordlist:      # 名詞句   apple  microsoft  steve jobs
        for word2 in clean_wordlist: # 固有名詞 apple  microsoft  steve jobs
            similarity = word_similarity(word,word2) # 
            # 類似度が0.12以上のペアを結合
            if similarity < 0.12:
                pass # apple,microsoft=0  apple,steve jobs=0  microsoft,apple=0  microsoft,microsoft=0  microsoft,steve jobs=0  steve jobs,apple=0  steve jobs,microsoft=0  steve jobs,steve jobs=0
            else:
                if word.lower() == word2.lower():
                    pass  # 同一単語は除外  apple,aple=1.0
                else:
                    out_wordlist.append(word+word2) # 重複する単語を格納しないよう変更

    return out_wordlist # []  []    ['appleorange', 'orangeapple']  ['appleorange', 'orangeapple']

# ========================================
# 関数: Wikipediaから年を抽出
# ========================================         
def get_year(proper_noun): # apple  steve  jobs  microsoft  orange  redmond  nokia  washington
    """
    固有名詞（人名等）のWikipediaページから西暦を抽出
    
    例: "Einstein" → "1879" (生年)
    
    パスワードに生年月日を使うユーザーが多いため、
    関連する年を抽出してパスワード候補に追加
    """
    try:
        page = wikipedia.page(proper_noun) # オンライン用  オフラインのときrequests.exceptions.ConnectionError:で下に進まない
        # 1000〜3999年の4桁数字を抽出（実行ごとに年を取得できる単語が変わる場合がある）
        year = re.findall('[1-3][0-9]{3}',page.summary) # []  []  []  ['1975', '1980', '1980', '2021', '1986', '1990', '2000', '2011', '2012', '2014', '2016', '2022', '2023', '1990', '2019']  []  ['1086', '2009'] []  []
#       print(proper_noun, year) # steve ['2016']
                                 # microsoft ['1975', '1980', '1980', '2021', '1986', '1990', '2000', '2011', '2012', '2014', '2016', '2022', '2023', '1990', '2019']
                                 # gaza ['1943', '1975', '1997']          
                                 # redmond ['1086', '2009']
        return year[0] # microsoftのとき['1975']  redmondのとき['1086']
    except:
        return None # apple  steve  jobs  orange  nokia  washington

# ========================================
# 関数: Wikipediaから地名抽出
# ========================================
def get_cities(proper_noun): # apple  steve  jobs
    """
    固有名詞のWikipediaページから都市名を抽出
    
    例: "Tom Brady" → ["Boston", "Tampa"]
    
    出身地や居住地など、本人に関連する地名を
    パスワード候補として収集
    """
    page = wikipedia.page(proper_noun) # オンライン用  オフラインのときrequests.exceptions.ConnectionError:で下に進まない
    geo = GeoText(page.summary) # <geotext.geotext.GeoText object at 0x0000018DC5787190>
    return list(set(geo.cities))

# ========================================
# 関数: 正規表現から文字列生成
# ========================================     
def regex_parser(regex): # [0-9]2
    """
    正規表現パターンから実際の文字列を生成
    
    例: "[0-9]{2}" → ["00", "01", "02", ... "99"]
    
    パスワードの末尾によく付く数字パターン等を
    正規表現で指定して一括生成
    """
    try:
        string_list = list(exrex.generate(regex)) # ['02', '12', '22', '32', '42', '52', '62', '72', '82', '92']
        return string_list
    except:
        print("Incorrect regex syntax")
        sys.exit(1)

# ========================================
# 関数: マスクパターン適用
# ========================================
def mask_parser(mask,word): # ?u apple    ?u?u apple
    """
    単語に大文字/小文字のマスクパターンを適用
    
    例: mask="?u?l?l", word="dog" → "Dog"
        ?u = 大文字, ?l = 小文字
    
    パスワードポリシーで「先頭大文字」等が
    要求される場合に対応
    """
    mask_items = mask.split('?')  # ['', 'u']    ['', 'u', 'u']
    mask_length = len(mask_items) # 2    3
    word_chars = list(word)       # ['a', 'p', 'p', 'l', 'e']
    i = 0
    for mask in mask_items[1:]:   # u
        if len(word_chars) < i+1:
            break
        else:    
            if mask == 'l':
                word_chars[i] = word_chars[i].lower()
            elif mask == 'u':
                word_chars[i] = word_chars[i].capitalize()
            i += 1
    final_word = ''.join(word_chars) # Apple    APple
    return final_word

# ========================================
# 関数: ファイル行数カウント
# ========================================             
def file_len(fname): # none_wordlist.txt
    """生成された単語リストの行数を数える"""
    with open(fname) as f:
        for i, l in enumerate(f): # l=orangeapple1086
            pass
    return i + 1 # 47 + 1  最終的な行数は48

# ========================================
# メイン処理関数
# ======================================== 
def action(twitter_username,word_type='all',regex_place='suffix',regex=None,mask=None,filename=None,urlfile=None):
    """  # twitter_username=none
    メインのパスワードリスト生成処理
    
    処理フロー:
    1. Twitter/ファイル/URLからテキスト取得
    2. NLPで名詞句・固有名詞を抽出
    3. 類似単語ペアを生成
    4. Wikipedia連携で年・地名を追加
    5. 正規表現・マスクを適用
    6. ファイルに出力
    """
    final_wordlist = []
    year_list = []
    city_list = []
    noun_phrases_display = []
    proper_nouns_display = []
    text = ''
    
    # ========================================
    # 入力ソースの選択と読み込み
    # ========================================
#   if twitter_username is not "none": 変更
    if twitter_username != "none": # twitter_username = none
        # Twitter から取得 来ない
        try:
            print("Downloading tweets from: " + twitter_username) # 次の場所からツイートをダウンロードしています:
            text = get_all_tweets(twitter_username) # get_all_tweets関数へ 
        except:
            print("Couldn't download tweets. Your credentials maybe wrong or you rate limited.") # ツイートをダウンロードできませんでした。認証情報が間違っているか、レート制限が適用されている可能性があります。
            sys.exit(1)
        print("Analyzing tweets, this will take a while..") # ツイートを分析中です。しばらく時間がかかります。
        print("")
    else:
        # ファイルまたはURLから取得
        if filename is None and urlfile is None:
            print("No input source is specified") # 入力ソースが指定されていません 来ない
            sys.exit(0) # 終了
        elif filename is not None and urlfile is None: # filename=mydata.txt  urlfile=None
            # テキストファイルから読み込み
            print("Analyzing the text file..")
            print("")
            try:
#               with open(filename, 'r') as textfile: 変更          
                with open(filename, 'r', encoding='utf-8', errors='ignore') as textfile: # テキストファイルを読み込む
                    text = textfile.read().replace('\n', ' ') # Apple was founded by Steve Jobs. Microsoft is another tech giant.
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1) # 終了
        elif filename is None and urlfile is not None:
            # URL一覧ファイルから各URLの内容を取得 来ない
            print("Analyzing the given URLs..")
            print("")
            with open(urlfile, 'r') as textfile:    # urlfile=blogs.txt(urlだけの行、スペースなどで分割はされない)
                urls = textfile.read().splitlines() # ['https://example.com/post1.html https://example.com/post2.html https://cnn.com/news.html https://en.wikipedia.org/wiki/Censorate']
                for url in urls:
                    print("Connecting to: " + url)  # Connecting to: https://example.com/post1.html
                    try: 
                        r = requests.get(url,timeout=20) # <Response [404]> 指定した URL に対応するリソースが存在しない              https://example.com/post1.html https://example.com/post2.html https://cnn.com/news.html
                    except:                              # <Response [403]> サーバーがリクエストを理解したが、アクセス権限がないため拒否された URL(https://en.wikipedia.org/wiki/Censorate)のとき   https://ja.wikipedia.org/robots.txt  https://www.yahoo.co.jp/robots.txt
                        print("Connection failed") # urlが123など無効なアドレスの時
                        continue
                    # HTMLから本文を抽出（スクリプト・スタイルは除外、空行も排除）
                    soup = BeautifulSoup(r.text)
                    for script in soup(["script", "style"]):
                        script.decompose()
                    clean_text = soup.get_text()                               # Example DomainExample DomainThis domain is for use in documentation examples without needing permission. Avoid use in operations.Learn more
                    lines = (line.strip() for line in clean_text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    clean_text = '\n'.join(chunk for chunk in chunks if chunk) # Example DomainExample DomainThis domain is for use in documentation examples without needing permission. Avoid use in operations.Learn more
                                                                               # Please set a user-agent and respect our robot policy https://w.wiki/4wJS. See also https://phabricator.wikimedia.org/T400119.
                    text += clean_text 
    
    # ASCII文字のみに制限（非ASCII文字を除去） 単語の中に非ASCII文字がある場合は、非ASCII文字だけ除去する
    text = ''.join(i for i in text if ord(i)<128) # Apple was founded by Steve Jobs. Microsoft is another tech giant.
    
    # ========================================
    # NLP解析: 名詞句と固有名詞の抽出
    # ========================================
    noun_phrases = find_noun_phrases(text) # find_noun_phrases関数へ 名詞句（noun phrases）の抽出   
                                           # [('apple', 2), ('microsoft', 1), ('steve jobs', 0)]
    print("\n名詞句（noun phrases）", noun_phrases)
    
    proper_nouns = find_proper_nouns(text) # find_proper_nouns関数へ 固有名詞（proper nouns）の抽出 
                                           # [('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1)]
    print("\n固有名詞（proper nouns）", proper_nouns)
    
    # 追加 Noneチェック
    if noun_phrases is None:
        noun_phrases = []
    if proper_nouns is None:
        proper_nouns = []
    
    # ========================================
    # 類似単語ペアの生成
    # ========================================
    paired_nouns = mass_similarity_compare(noun_phrases)   # mass_similarity_compare関数へ []  
                                                           # ['appleorange', 'orangeapple']
    print("\n名詞句（noun phrases）類似単語ペアの生成", noun_phrases, paired_nouns)
    
    paired_propers = mass_similarity_compare(proper_nouns) # mass_similarity_compare関数へ []  
                                                           # ['appleorange', 'orangeapple']
    print("\n固有名詞（proper nouns）類似単語ペアの生成", proper_nouns, paired_propers)
    print("") 
    
    # ========================================
    # 結果表示（色付き）
    # ========================================
    for i in range(len(noun_phrases)): # [('apple', 1), ('microsoft', 1), ('orange', 1), ('steve jobs', 0)]
        noun_phrases_display.append(str(noun_phrases[i][0])+":"+str(noun_phrases[i][1])) # ['apple:1', 'microsoft:1', 'orange:1', 'steve jobs:0']
       
    print(Fore.GREEN + 'Most used nouns: ' + Style.RESET_ALL + ", ".join(noun_phrases_display))
                      # Most used nouns: apple:1, microsoft:1, orange:1, steve jobs:0

    for i in range(len(proper_nouns)): # [('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1), ('orange', 1)]
        proper_nouns_display.append(str(proper_nouns[i][0])+":"+str(proper_nouns[i][1])) # ['apple:1', 'steve:1', 'jobs:1', 'microsoft:1', 'orange:1']
       
    print((Fore.GREEN + 'Most used proper nouns: ' + Style.RESET_ALL + ", ".join(proper_nouns_display)))
                       # Most used proper nouns: apple:1, steve:1, jobs:1, microsoft:1, orange:1
    print("") 
    
    # ========================================
    # Wikipedia連携: 関連地名の収集
    # ========================================
    print("Gathering related locations and years..") # 関連する場所と年を収集します。
    print("")
    for noun in proper_nouns: # ('apple', 1) ('steve', 1) ('jobs', 1) ('microsoft', 1) ('orange', 1)
#       print("Getting info for: " + str(noun)) # Getting info for: ('apple', 1)
        try:
            # 固有名詞ごとにWikipediaから最大3都市を取得
            temp_city_list = get_cities(noun[0])[0:3] # Wikipedia + GeoText使用 オンライン用 get_cities関数へ 
                                                      # ('microsoft', 1)のとき、['Redmond', 'Washington', 'Nokia']               
        except: # wikipedia.exceptions.DisambiguationError(検索した語が曖昧で複数の候補ページに該当するときに発生する例外。「どのページを返せばいいか分からない」という状況を知らせるためのエラー)
            continue # jobs  orange  オフラインのとき
        
        if(temp_city_list != []): # 追加
            print("wiki", noun, temp_city_list)
            
        for city in temp_city_list: # Redmond  Washington  Nokia
            city_list.append(city.replace(" ", "").lower()) # ['redmond', 'washington', 'nokia']
        
        
    # 重複を除去して固有名詞リストに追加
    if city_list: # ['redmond', 'washington', 'nokia']
        city_list = list(set(city_list)) # ['redmond', 'nokia', 'washington']
        for city in city_list:
            city_tuple = (city,0)           # ('redmond', 0)  ('nokia', 0)  ('washington', 0)
            proper_nouns.append(city_tuple) # [('apple', 1), ('steve', 1), ('jobs', 1), ('microsoft', 1), ('orange', 1), ('redmond', 0), ('nokia', 0), ('washington', 0)]
    
    # ========================================
    # 単語リストの構築
    # ========================================
    
    # 1. 名詞句を追加
    for word in noun_phrases: # ('apple', 1)  ('microsoft', 1)  ('orange', 1)  ('steve jobs', 0)
        if mask is None:      # None
            final_wordlist.append(word[0]) # ['apple', 'microsoft', 'orange', 'steve jobs']
        else:
            final_wordlist.append(mask_parser(mask,word[0])) # mask=?u
    '''
    # 名詞句を分割したものも使用、大文字、1文字目だけ大文字も追加
    for word in noun_phrases:
        word_str = word[0]
    
        if mask is None:
            # 元の複合語
            final_wordlist.append(word_str)
        
            # 複合語を分割（スペース区切り）
            words_split = word_str.split()
            if len(words_split) > 1:  # 複数単語の場合のみ
                for single_word in words_split:
                    if len(single_word) >= 3:  # 3文字以上
                        final_wordlist.append(single_word)
        
            # 大文字バリエーション
            final_wordlist.append(word_str.upper())  # 全て大文字
            final_wordlist.append(word_str.capitalize())  # 1文字目だけ大文字
        
            # 分割した単語の大文字バリエーション
            if len(words_split) > 1:
                for single_word in words_split:
                    if len(single_word) >= 3:
                        final_wordlist.append(single_word.upper())
                        final_wordlist.append(single_word.capitalize())
        else:
            final_wordlist.append(mask_parser(mask, word_str))
    '''    
    
    # 2. 固有名詞を追加（Wikipedia連携で年も取得、オフラインでは都市名なし）
    for word in proper_nouns: # ('apple', 1)  ('steve', 1)  ('jobs', 1)  ('microsoft', 1)  ('orange', 1)  ('redmond', 0)  ('nokia', 0)  ('washington', 0)
        if mask is None:
            final_wordlist.append(word[0]) # ['apple', 'microsoft', 'orange', 'steve jobs', 'apple', 'steve', 'jobs', 'microsoft', 'orange', 'redmond', 'nokia', 'washington']
#            final_wordlist.append(word[0].upper())  # 全て大文字
#            final_wordlist.append(word[0].capitalize()) # 1文字目だけ大文字            
        else:
            final_wordlist.append(mask_parser(mask,word[0]))
        
        # baseモード以外では年も収集
        if word_type != 'base': # 年付き単語（オフラインでは0個） all
            try:
                year = get_year(word[0]) # get_year関数へ None  None  None  1975
                if year != None:
                    print(word[0], year)
                    if year not in year_list: # year_listにyearがないとき
                        year_list.append(year) # ['1975', '1086']
            except:
                pass
    
    # 3. 類似単語ペアを追加
    for word in paired_nouns: # appleorange  orangeapple
        if mask is None:
            final_wordlist.append(word) # ['apple', 'microsoft', 'orange', 'steve jobs', 'apple', 'steve', 'jobs', 'microsoft', 'orange', 'redmond', 'nokia', 'washington', 'appleorange', 'orangeapple']
#            final_wordlist.append(word.upper())
#            final_wordlist.append(word.capitalize())            
        else:
            final_wordlist.append(mask_parser(mask,word))
    
    for word in paired_propers:
        if mask is None:
            final_wordlist.append(word) # ['apple', 'microsoft', 'orange', 'steve jobs', 'apple', 'steve', 'jobs', 'microsoft', 'orange', 'redmond', 'nokia', 'washington', 'appleorange', 'orangeapple', 'appleorange', 'orangeapple']
#            final_wordlist.append(word.upper())
#            final_wordlist.append(word.capitalize())  
        else:
            final_wordlist.append(mask_parser(mask,word))

    # 追加　重複を削除
    final_wordlist = list(set(final_wordlist))
    
    # ========================================
    # 正規表現パターンの適用
    # ========================================
    if regex_place is not None or regex is not None: # none    suffix [0-9]2  {}はなくなる
        new_items = regex_parser(regex) # regex_parser関数へ  ['02', '12', '22', '32', '42', '52', '62', '72', '82', '92']
        with open("regex_words.txt",'w+') as regex_words: # a+からw+に変更
            for item in new_items: # 02
                for word in final_wordlist: # apple
#               with open("regex_words.txt","a+") as regex_words:
                    # prefix: 先頭に追加、suffix: 末尾に追加
                    if regex_place == 'prefix':
                        regex_words.write(item+word+'\n')
                    else:
                        regex_words.write(word+item+'\n') # apple02  microsoft02 ・・・ 
       
    # ========================================
    # ファイル出力
    # ========================================
    with open(twitter_username+"_wordlist.txt",'w+') as wordlist:
        # 基本単語を出力
        for word in final_wordlist: # apple  microsoft  orange  steve jobs  apple  steve  jobs  microsoft  orange  redmond  nokia  washington  appleorange  orangeapple  appleorange  orangeapple
            wordlist.write(word+'\n')
        
        # baseモード以外では年付き単語も出力
        if word_type != 'base': # all
            for word in final_wordlist:
                for year in list(set(year_list)): # 1975  1086
                    wordlist.write(word+year+'\n')
    
    # 正規表現で生成した単語を結合(Windowsではcatでエラーになるためコメントアウト)
#    if regex is not None: # none 
#        os.system('cat regex_words.txt >> ' +twitter_username+"_wordlist.txt") # 'cat' is not recognized as an internal or external command, operable program or batch file.
#        os.remove('regex_words.txt') # regex_words.txtを削除
    
    # ========================================
    # 結果レポート
    # ========================================
    raster_size = os.path.getsize(twitter_username+"_wordlist.txt") # 599  ファイルサイズを取得
    print("")
    print("Wordlist is written to: " + twitter_username+"_wordlist.txt") # Wordlist is written to: none_wordlist.txt
    print("出力ファイルのサイズ: " + size(raster_size, system=alternative)) # Size of the wordlist: 599 bytes
    print("出力ファイルの行数: " + str(file_len(twitter_username+"_wordlist.txt"))) # file_len関数へ  Number of lines in wordlist: 48

# ========================================
# コマンドライン引数の解析
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--username', action='store', dest='username', help='Twitter username', required=False,default="none")
parser.add_argument('--regex_place', action='store', dest='regex_place', help='Regex place: prefix or suffix', required=False)
parser.add_argument('--regex', action='store', dest='regex', help='Regex syntax', required=False)
parser.add_argument('--mask', action='store', dest='mask', help='Mask structure of wordlist', required=False)
parser.add_argument('--filename', action='store', dest='filename', help='Arbitrary textfile to analyze', required=False) # 分析する任意のテキストファイル
parser.add_argument('--urlfile', action='store', dest='urlfile', help='File which contains URLs to analyze', required=False) # 分析するURLを含むファイル
#argv = parser.parse_args()
# --regex '[0-9]{4}'    --regex "(2020|2021|2022)"   --regex '[!@#$]'  -regex '[0-9]{1,3}'  --regex_place suffix  --mask "?u"  --mask "?u?l?l?l?l"
import traceback
try:
    argv = parser.parse_args()
except SystemExit as e:
    traceback.print_exc()
    print(f"sys.argv: {sys.argv}")
#print(argv) # Namespace(username='none', regex_place=None, regex=None, mask=None, filename=None, urlfile=None)

# ========================================
# メイン実行
# ========================================
action(twitter_username=argv.username,regex_place=argv.regex_place,regex=argv.regex,mask=argv.mask,filename=argv.filename,urlfile=argv.urlfile)