import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import math
import pandas as pd
import os

# 1. 업로드된 새로운 동물 판별 모델과 라벨을 안전하게 캐싱
@st.cache_resource
def load_keras_model():
    if os.path.exists("keras_model (1).h5"):
        return load_model("keras_model (1).h5", compile=False)
    elif os.path.exists("model/keras_model (1).h5"):
        return load_model("model/keras_model (1).h5", compile=False)
    else:
        return load_model("../keras_model (1).h5", compile=False)

@st.cache_resource
def load_labels():
    if os.path.exists("labels (1).txt"):
        path = "labels (1).txt"
    elif os.path.exists("model/labels (1).txt"):
        path = "model/labels (1).txt"
    else:
        path = "../labels (1).txt"
    return open(path, "r", encoding='UTF-8').readlines()


# 📍 이미지에서 GPS 위치 정보를 추출하는 함수
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


# 📸 디렉토리에 파일, 위치 데이터, 판별된 동물 종류를 함께 저장하는 함수
def save_uploaded_file_with_gps(directory, file, gps_data, animal_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # 1. 원본 이미지 파일 저장
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
        
    # 2. 개별 위치 정보 텍스트 파일 저장 (종류 데이터 추가 반영)
    base_name, _ = os.path.splitext(file.name)
    txt_path = os.path.join(directory, f"{base_name}_location.txt")
    
    if gps_data:
        lat, lng = gps_data
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Latitude (위도): {lat}\nLongitude (경도): {lng}\nDetected Animal (판별 종류): {animal_name}\n")
        
        # 3. 📊 근처 탐색을 위한 통합 데이터베이스(CSV)에 누적 저장
        csv_path = os.path.join(directory, "location_history.csv")
        new_data = pd.DataFrame([{"lat": lat, "lon": lng, "animal": animal_name, "file_name": file.name}])
        
        if os.path.exists(csv_path):
            history_df = pd.read_csv(csv_path)
            # 중복 이미지 저장을 방지하기 위해 파일명이 다를 때만 결합
            if file.name not in history_df['file_name'].values:
                history_df = pd.concat([history_df, new_data], ignore_index=True)
                history_df.to_csv(csv_path, index=False)
        else:
            new_data.to_csv(csv_path, index=False)
            
        st.success(f'Saved file & GPS metadata with animal class ({animal_name}) in {directory}')
    else:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"No GPS data found in this image. Detected Animal: {animal_name}\n")
        st.warning(f'Saved file in {directory} (위치 데이터 없음)')


# 🔍 현재 위치 근처(오차범위 약 1~2km 이내)에서 발견된 적 있는 과거 종 정보 조회 함수
def get_nearby_history(directory, current_gps):
    csv_path = os.path.join(directory, "location_history.csv")
    if not os.path.exists(csv_path) or current_gps is None:
        return []
    
    try:
        history_df = pd.read_csv(csv_path)
        cur_lat, cur_lon = current_gps
        
        # 📍 위도/경도 오차가 ±0.015 이내인 경우 근처(반경 약 1.5km 이내)로 판단
        threshold = 0.015
        nearby_condition = (abs(history_df['lat'] - cur_lat) <= threshold) & (abs(history_df['lon'] - cur_lon) <= threshold)
        nearby_records = history_df[nearby_condition]
        
        # 발견되었던 동물 종류 목록만 추출 (중복 제거)
        past_animals = nearby_records['animal'].unique().tolist()
        return past_animals
    except Exception:
        return []


# 🔮 새로운 동물 판별 모델의 예측을 수행하는 함수
def predict_animal(image, model, class_names):
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
    st.title('에일리언 GO🐾🌿')
    st.info('낯선 동식물의 이미지를 업로드하고 포인트를 받아가세요!')
    
    image = st.file_uploader('촬영한 사진을 업로드해주세요!', type=['jpg','png','jpeg','webp'])

    if image is not None:
        input_image = Image.open(image)
        
        # 📍 1. 위치 데이터 추출 및 사이드바 지도 표시
        gps_data = get_gps_info(input_image)
        
        if gps_data:
            st.sidebar.marker_data = pd.DataFrame({'lat': [gps_data[0]], 'lon': [gps_data[1]]})
            st.sidebar.write(f"📍 촬영 위치 발견! (위도: {round(gps_data[0], 4)}, 경도: {round(gps_data[1], 4)})")
            st.sidebar.map(st.sidebar.marker_data)
        
        st.image(input_image, caption='업로드된 이미지', use_container_width=True)

        # 📍 2. 새로운 모델 및 라벨 로딩 후 판별
        model = load_keras_model()
        class_names = load_labels()
        animal_name, confidence, all_predictions = predict_animal(input_image, model, class_names)

        # 📍 3. [신규 기능] 이미지 분석 후, 해당 위치 반경의 과거 기록 가져오기
        save_directory = "uploaded_images"
        past_nearby_animals = get_nearby_history(save_directory, gps_data)

        # 📍 4. 사진과 위치 데이터 + 동물 종류를 같이 저장하기
        save_uploaded_file_with_gps(save_directory, image, gps_data, animal_name)

        # 분석 결과 화면 출력
        st.success(f'분석 결과: 해당 동물은 현재 {math.floor(confidence*100*10)/10}% 확률로 **[{animal_name}]** 입니다.\n(100포인트가 지급되었습니다.)')

        # 📍 5. [신규 기능] 로케이션 정보가 있고 과거 발견 데이터가 존재하면 추가로 언급해주기
        if gps_data:
            st.markdown("---")
            st.subheader("🗺️ 이 지역 주변 과거 관측 정보")
            if past_nearby_animals:
                # 리스트 내용을 콤마(,)로 연결하여 보여주기
                animals_str = ", ".join([f"**[{a}]**" for a in past_nearby_animals])
                st.info(f"💡 현재 촬영 장소 주변에서 과거에 {animals_str}이(가) 발견된 기록이 있습니다! 주의하세요!")
            else:
                st.write("🔍 이 장소 근처에서 과거에 기록된 다른 동물 정보가 아직 없습니다. 첫 발견 기록입니다!")

        # 📊 확률 분포 막대그래프 시각화
        st.subheader('📊 동물별 확률 분포')
        animal_df = pd.DataFrame({
            'animal': [name[2:].strip() for name in class_names], 
            'probability': all_predictions
        })
        st.bar_chart(animal_df.set_index('animal'))

if __name__=='__main__':
    main()