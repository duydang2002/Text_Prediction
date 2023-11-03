import re
import unicodedata
import string
import random
import nltk
#nltk.download()
from nltk.probability import ConditionalFreqDist


def main():
    # Đọc Dữ liệu từ file text với bộ mã hóa kí tự utf-8
    text = ""
    with open("./alice.txt", encoding="utf-8") as file:
        
        while True:
            line = file.readline()
            text += line
            if not line:
               break
              
    # Tiền xử lí dữ liệu
    print("Filtering...")
    words = filter(text)
    print("Cleaning...")
    words = clean(words)
    # Tạo model
    print("Making model...")
    model = n_gram_model(words)

    print("Enter a phrase: ")
    user_input = input()
    predict(model, user_input)


"""
    Normalize text, remove unnecessary characters, 
    perform regex parsing, and make lowercase
"""
def filter(text):
    # Trực chuẩn hóa các kí tự theo kiểu NFKD - tách dấu và kí tự ra riêng
    # sau đó mã hóa các kí tự thành các kí tự ascii, kí tự nào không encode được thì bỏ qua
    # Cuối cùng là decode nó lại dạng chuẩn utf-8 và cũng bỏ qua các kí tự không decode được
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    
    # thay thế các thẻ HTML nếu có thành ' '
    # . là bất kì kí tự nào khác xuống dòng, *? là match 0 hay nhiều kí tự lặp lại 
    # sao cho số kí tự ít nhất có thể
    
    text = re.sub('<.*?>', ' ', text)
    # Xóa dấu chấm câu
    # string.punctuation là !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # str.maketrans để tạo ra 1 table mà thay kí tự '' thành ' ' và xóa các kí tự string.punctuation
    
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    
    # Thay thế các kí tự nào không phải là chữ cái hay chữ hoa thành ' '
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Thay thế kí tự xuống dòng bằng dấu cách
    text = re.sub("\n", " ", text)
    
    # lower case
    text = text.lower()
    
    # Dùng split để tách text ra bằng dấu ' ' ở đây cũng sẽ bỏ hết các dấu cách thừa 
    # rồi lại dùng ' ' để ghép các kí tự này lại nên ta sẽ xóa được hết các dấu cách thừa
    text = ' '.join(text.split())
    return text

"""
    Tokenize remaining words
    and perform lemmatization
"""
def clean(text):
    # Tách text ra thành các từ riêng lẻ
    tokens = nltk.word_tokenize(text)
    # Sử dụng WordNet Lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()

    output = []
    for words in tokens:
        # Rút gọn các từ có các form khác nhau thành 1 form ví dụ Building, Builds Built thành Build
        output.append(wnl.lemmatize(words))
    return output


"""
    Make a language model using a dictionary, trigrams, 
    and calculate word probabilities
"""
def n_gram_model(text):
    # Tạo ra một list 3-grams từ text
    # Nếu không đủ 3 kí tự thì sẽ pad_left và pad_right bằng kí tự <s>
    trigrams = list(nltk.ngrams(text, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

    # bigrams = list(nltk.ngrams(text, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

# N-gram Statistics
    # get freq dist of trigrams
    # freq_tri = nltk.FreqDist(trigrams)
    # freq_bi = nltk.FreqDist(bigrams)
    # freq_tri.plot(30, cumulative=False)
    # print("Most common trigrams: ", freq_tri.most_common(5))
    # print("Most common bigrams: ", freq_bi.most_common(5))

    # Tạo một dictionary mà lưu trữ tần suất xuất hiện của kí tự thứ 3 trên 2 kí tự trước nó 
    # ở trong trigrams
    cfdist = ConditionalFreqDist()
    for w1, w2, w3 in trigrams:
        cfdist[(w1, w2)][w3] += 1

    # Chuyển đổi tần suất thành xác suất
    # Đếm tổng các tần suất rồi lấy tần suất của 1 trường hợp cụ thể chia cho tổng đó
    for w1_w2 in cfdist:
        total_count = float(sum(cfdist[w1_w2].values()))
        for w3 in cfdist[w1_w2]:
            cfdist[w1_w2][w3] /= total_count

    return cfdist

"""
    Generate predictions from the Conditional Frequency Distribution
    dictionary (param: model), append weighted random choice to user's phrase,
    allow option to generate more words following the prediction
"""
def predict(model, user_input):
    user_input = filter(user_input)
    user_input = user_input.split()
    
    # Lấy prev word là 2 từ cuối của xâu input
    w1 = len(user_input) - 2
    w2 = len(user_input)
    prev_words = user_input[w1:w2]
    
    # Tạo ra một từ điển là xác suất xuất hiện kí tự thứ 3 dựa trên 2 kí tự trước nó 2 kí tự đứng trước
    # Sắp xếp theo xác suất xuất hiện
    prediction = sorted(dict(model[prev_words[0], prev_words[1]]), key=lambda x: dict(model[prev_words[0], prev_words[1]])[x], reverse=True)
    print("Trigram model predictions: ", prediction)

    word = []
    weight = []
    for key, prob in dict(model[prev_words[0], prev_words[1]]).items():
        word.append(key)
        weight.append(prob)
    # pick from a weighted random probability of predictions
    next_word = random.choices(word, weights=weight, k=1)
    # add predicted word to user input
    user_input.append(next_word[0])
    print(' '.join(user_input))

    ask = input("Do you want to generate another word? (type 'y' for yes or 'n' for no): ")
    if ask.lower() == 'y':
        predict(model, str(user_input))
    elif ask.lower() == 'n':
        print("done")
        

main()
