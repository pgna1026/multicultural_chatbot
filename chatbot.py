import os
from decouple import config
import openai
from openai.error import OpenAIError  # OpenAIError만 임포트
import pyttsx3  # TTS 기능을 위한 라이브러리
import speech_recognition as sr  # 음성 인식을 위한 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수에서 API 키 로드
openai.api_key = config('OPENAI_API_KEY')

# pyttsx3 엔진 초기화
engine = pyttsx3.init()

# 음성 인식을 위한 recognizer 객체 생성
recognizer = sr.Recognizer()

# 데이터셋 로드 및 TF-IDF 모델 생성
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def build_tfidf_model(data):
    vectorizer = TfidfVectorizer().fit(data)
    tfidf_matrix = vectorizer.transform(data)
    return vectorizer, tfidf_matrix, data

def retrieve_knowledge(query, vectorizer, tfidf_matrix, data):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = similarities.argsort()[-1]  # 가장 유사한 텍스트의 인덱스
    return data[top_idx]

# 사전 지식을 추가한 GPT-3.5 호출
def generate_response_with_rag(query, knowledge):
    prompt = f"Use the following knowledge:\n\n{knowledge}\n\nNow, respond to the user's query: {query}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful consultant for multicultural."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        
        return response.choices[0].message['content'].strip()

    except OpenAIError as e:
        return f"An error occurred: {e}"

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"User (via speech): {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

# 상담원과 대화를 생성하는 함수
def chat_with_assistant():
    conversation_history = [
        {"role": "system", "content": "You are a helpful consultant for multicultural."}
    ]

    # 데이터셋 로드 및 TF-IDF 모델 구축
    dataset_path = 'dataset.txt'
    data = load_dataset(dataset_path)
    vectorizer, tfidf_matrix, data = build_tfidf_model(data)

    # 프로그램 실행 시 모드 한 번만 설정
    mode = None
    while mode not in ['0', '1']:
        print("Select mode: 0 for typing, 1 for speaking")
        mode = input("Mode (0/1): ").strip()

        if mode not in ['0', '1']:
            print("Invalid mode selected. Please choose 0 (typing) or 1 (speaking).")

    while True:
        if mode == '1':
            user_input = recognize_speech()
        elif mode == '0':
            user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        # Knowledge Retrieval
        knowledge = retrieve_knowledge(user_input, vectorizer, tfidf_matrix, data)

        # GPT-3.5 with RAG
        assistant_reply = generate_response_with_rag(user_input, knowledge)
        print(f"Assistant: {assistant_reply}\n")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # TTS로 음성 출력
        engine.say(assistant_reply)
        engine.runAndWait()

# 상담원과 대화 시작
chat_with_assistant()