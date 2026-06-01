import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS  # 🔥 GPS 데이터 분석을 위한 모듈 추가
import numpy as np
import math
import pandas as pd
import os

# 1. 모델과 라벨을 안전하게 캐싱
@st.cache_resource
def load_keras_model():
    return load_model("model/keras_model.h5", compile=False)

@st.cache_resource
def load_labels():
    return open("model/labels.txt", "r", encoding='UTF-8').readlines()


# 📍 [새로운 함수] 이미지에서 GPS 위치 정보를 추출하는 함수
def get_gps_info(image):
    try:
        exif = image._getexif()
        if not exif:
            return None
        
        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'GPSInfo':
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            # 도, 분, 초 계산을 실수형(Degree) 데이터로 변환
            lat_ref = gps_info.get('GPSLatitudeRef', 'N')
            lng_ref = gps_info.get('GPSLongitudeRef', 'E')
            
            lat = gps_info['GPSLatitude']
            lng = gps_info['GPSLongitude']
            
            lat_deg = float(lat[0]) + float(lat[1])/60.0 + float(lat[2])/3600.0
            lng_deg = float(lng[0]) + float(lng[1])/60.0 + float(lng[2])/3600.0
            
            if lat_ref == 'S': lat_deg = -lat_deg
            if lng_ref == 'W': lng_deg = -lng_deg
            
            return lat_deg, lng_deg
    except Exception:
        return None
    return None


# 📸 디렉토리에 파일과 위치 데이터를 함께 저장하는 함수
def save_uploaded_file_with_gps(directory, file, gps_data):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # 1. 원본 이미지 파일 저장
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
        
    # 2. 위치 데이터가 있다면 텍스트 파일로 함께 저장
    base_name, _ = os.path.splitext(file.name)
    txt_path = os.path.join(directory, f"{base_name}_location.txt")
    
    if gps_data:
        lat, lng = gps_data
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Latitude (위도): {lat}\nLongitude (경도): {lng}\n")
        st.success(f'Saved file & GPS metadata in {directory}')
    else:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("No GPS data found in this image.\n")
        st.warning(f'Saved file in {directory} (위치 데이터 없음)')


# 2. 감정 예측을 수행하는 함수
def predict_emotion(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

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
        # AI 분석 및 GPS 추출을 위해 PIL 이미지로 열기
        input_image = Image.open(image)
        
        # 📍 위치 데이터 추출하기
        gps_data = get_gps_info(input_image)
        
        if gps_data:
            st.sidebar.marker_data = pd.DataFrame({'lat': [gps_data[0]], 'lon': [gps_data[1]]})
            st.sidebar.write(f"📍 촬영 위치 발견! (위도: {round(gps_data[0], 4)}, 경도: {round(gps_data[1], 4)})")
            # 스트림릿 내장 지도로 사이드바에 촬영 장소 표시하기
            st.sidebar.map(st.sidebar.marker_data)
        
        # 📍 사진과 위치 데이터(.txt) 함께 저장하기
        save_uploaded_file_with_gps("uploaded_images", image, gps_data)

        st.image(input_image, caption='업로드된 이미지', use_container_width=True)

        # 모델 및 예측 구동
        model = load_keras_model()
        class_names = load_labels()
        emotion, confidence, all_predictions = predict_emotion(input_image, model, class_names)

        st.success(f'반려견은 현재 {math.floor(confidence*100*10)/10}% 확률로 **[{emotion}]** 상태입니다.')

        st.subheader('📊 감정 확률 분포')
        emotion_df = pd.DataFrame({
            'emotion': [name[2:].strip() for name in class_names], 
            'probability': all_predictions
        })
        st.bar_chart(emotion_df.set_index('emotion'))

if __name__=='__main__':
    main()