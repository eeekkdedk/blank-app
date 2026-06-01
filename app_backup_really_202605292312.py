import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import math
import pandas as pd

# 1. 모델과 라벨을 안전하게 캐싱 (최신 표준 사용)
@st.cache_resource
def load_keras_model():
    return load_model("model/keras_model.h5", compile=False)

@st.cache_resource
def load_labels():
    return open("model/labels.txt", "r", encoding='UTF-8').readlines()

# 2. 감정 예측을 수행하는 함수
def predict_emotion(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # 이미지 정규화
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 예측 진행
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:].strip(), confidence_score, prediction[0]


def main():
    st.title('반려견의 감정을 알아보자! 🐕')
    st.info('반려견의 사진을 업로드 하면, 반려견의 표정을 분석하여 감정을 나타내줍니다.')
    
    image = st.file_uploader('반려견을 보여주세요!', type=['jpg','png','jpeg','webp'])

    if image is not None:
        image = Image.open(image)
        st.image(image, caption='업로드된 이미지', use_container_width=True)

        # 캐싱된 함수 호출
        model = load_keras_model()
        class_names = load_labels()

        # 감정 예측 결과 가져오기
        emotion, confidence, all_predictions = predict_emotion(image, model, class_names)

        # 결과 화면 표시
        st.success(f'반려견은 현재 {math.floor(confidence*100*10)/10}% 확률로 **[{emotion}]** 상태입니다.')

        # 감정 확률 시각화
        st.subheader('📊 감정 확률 분포')
        emotion_df = pd.DataFrame({
            'emotion': [name[2:].strip() for name in class_names], 
            'probability': all_predictions
        })
        st.bar_chart(emotion_df.set_index('emotion'))

if __name__=='__main__':
    main()