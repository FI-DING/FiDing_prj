from flask import Flask, jsonify, request, Response
from flask_cors import CORS ,cross_origin # CORS 추가
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import face_recognition
import pickle
import requests
import face_recognition_knn
import numpy as np
from sklearn.preprocessing import StandardScaler

from camera import VideoCamera

# PTY: 강수형태, REH: 습도( %), RN1: 1시간 강수량(mm), T1H: 기온(℃), UUU: 동서바람성분(m / s):, VEC: 풍향(deg), VVV: 남북바람성분(m / s), WSD: 풍속(m / s),
# MSRSTE_NM: 조회하는 구,
# NO2: 이산화질소농도(ppm), O3: 오존농도(ppm), CO: 일산화탄소농도(ppm), SO2: 아황산가스(ppm), PM10: 미세먼지(㎍ / ㎥), PM25: 초미세먼지(㎍ / ㎥)
# spdValue: 교통 속도, momentDateValue: localTime(현재 날짜와 시간)

app = Flask(__name__)
CORS(app)  # 모든 엔드포인트에 대한 CORS 활성화
# 더미 데이터
loaded_models = {}

#####################################deep 부분


# # ESP32 URL
# ESP32_URL = "http://192.168.0.121:81/stream"
#
# # Face recognition and OpenCV setup
# cap = cv2.VideoCapture(ESP32_URL)

def set_face_detect(url: str, face_detect: int=1):
    cap = cv2.VideoCapture(0)
    try:
        requests.get(url + "/control?var=face_detect&val={}".format(1 if face_detect else 0))
    except:
        print("SET_FACE_DETECT: something went wrong")
    return face_detect

def set_face_detect2(url: str, face_detect: int=1):
    try:
        requests.get(url + "/control?var=face_detect&val={}".format(1 if face_detect else 0))
    except:
        print("SET_FACE_DETECT: something went wrong")
    return face_detect
######################################
def pearson_correlation_coefficient(y_true, y_pred):
    mx = tf.reduce_mean(y_true)
    my = tf.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.reduce_sum(xm * ym)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm)) * tf.reduce_sum(tf.square(ym)))
    r = r_num / (r_den + tf.keras.backend.epsilon())
    return tf.reduce_mean(r)
def create_model(shape):
    model = tf.keras.Sequential([
        keras.layers.Dense(32, activation='sigmoid', input_shape=(shape,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(1)  # Output layer for regression task
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MSE,
                  metrics=[pearson_correlation_coefficient])
    return model

# def webcamStream():
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return None
#
#     window_name = "Live Webcam Feed"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     frames = []
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Error: Failed to grab frame.")
#             break
#
#         frames.append(frame)
#
#         cv2.imshow(window_name, frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return frames

# @app.route('/api/webcam-stream', methods=['GET'])
# @cross_origin()
# def get_data_webcam():
#     print("웹캠 시작.")
#     frames = webcamStream()
#
#     # Process frames or perform any additional logic if needed
#     print("웹캠 종료.")
#     return jsonify({"message": "Webcam stream ended."})

@app.route('/api/data', methods=['POST','GET'])
@cross_origin()
def get_data():
    api_data = request.get_json()
    result = {}
    m_list = ['아황산가스', '일산화탄소', '오존', '이산화질소', 'PM10', 'PM2.5']
    m_code_list = ["SO2", "CO", 'O3', 'NO2', 'PM10', 'PM25']

    try:
        api_data = request.get_json()
        for m, code in zip(m_list, m_code_list):
            mean, std = np.load(f'scaler_mean_std{m}.npy')
            scaler = StandardScaler()
            scaler.mean_ = mean
            scaler.scale_ = std
            if m in ['아황산가스', '일산화탄소', '오존', '이산화질소']:
                shape = 7
            else:
                shape = 4
            model = create_model(shape)
            model_path = f'C:\\Users\\jey92\\project\\team\\3차\\project3\\backend\\{m}_model_weights.h5'
            model.load_weights(model_path)
            print([[api_data['trafficData'], api_data['spdValue'], api_data['T1H'], api_data['WSD'], api_data['RN1'], api_data['REH'], float(api_data[code])]],api_data)
            if m in ['아황산가스', '일산화탄소', '오존', '이산화질소']:
                X=[[api_data['trafficData'], api_data['spdValue'], api_data['T1H'], api_data['WSD'], api_data['RN1'],api_data['REH'], float(api_data[code])]]
                X=scaler.transform(X)
                result[code] = float(model.predict(X)[0][0])
            else:
                X=[[api_data['trafficData'], api_data['spdValue'], api_data['T1H'], float(api_data[code])]]
                X = scaler.transform(X)
                result[code] = float(model.predict(X)[0][0])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/video_feed2') #웹캠으로 얼굴인식하는 부분
@cross_origin()
def video_feed2():

    print("웹캠함수실행")
    with open("trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)

    set_face_detect("http://localhost:5000", 1)

    cap = cv2.VideoCapture(0)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                face_locations = face_recognition.face_locations(frame)

                if len(face_locations) == 0:
                    continue

                faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
                closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(face_locations))]

                for (top, right, bottom, left), are_match in zip(face_locations, are_matches):
                    if are_match:
                        name = knn_clf.predict([faces_encodings[are_matches.index(True)]])[0]
                        print("인증성공",name)
                    else:
                        name = "unknown"
                        print("인증실패",name)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, bottom + 20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                _, jpeg = cv2.imencode('.jpg', frame)
                # response_data = {'name': name}

                return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() +
                                b'\r\n\r\n' + b'data: ' + name.encode('utf-8') + b'\r\n\r\n',
                                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video_feed3') #외장캠으로 얼굴인식하는 부분
@cross_origin()
def video_feed3():
    print("외장캠함수실행")
    with open("trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)

    set_face_detect2("http://localhost:5000", 1)
    # ESP32 URL
    ESP32_URL = "http://192.168.0.121:81/stream"

    # Face recognition and OpenCV setup
    cap = cv2.VideoCapture(ESP32_URL)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                face_locations = face_recognition.face_locations(frame)

                if len(face_locations) == 0:
                    continue

                faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
                closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(face_locations))]

                for (top, right, bottom, left), are_match in zip(face_locations, are_matches):
                    if are_match:
                        name = knn_clf.predict([faces_encodings[are_matches.index(True)]])[0]
                        print("인증성공",name)
                    else:
                        name = "unknown"
                        print("인증실패",name)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                _, jpeg = cv2.imencode('.jpg', frame)
                # response_data = {'name': name}

                return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n' + b'data: ' + name.encode('utf-8') + b'\r\n\r\n',
                                mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/api/data2') #강사님께서 주신 카메라로 얼굴을 인식하여 로그인 정보를 전달하는 함수
# @cross_origin()
# def get_data2():
#     data = request.json
#     with open("trained_knn_model.clf", 'rb') as f:
#         knn_clf = pickle.load(f)
#     result={}
#    # set_face_detect(URL, 1)
#     cap= cv2.VideoCapture(0)
#     while True:
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 # Find all face locations in the current frame
#                 face_locations = face_recognition.face_locations(frame)
#                 print(face_locations)
#                 # If no faces are found in the frame, continue to the next frame
#                 if len(face_locations) == 0:
#                     continue
#
#                 # Find face encodings for the faces in the frame
#                 faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
#
#                 # Use the KNN model to find the best matches for the faces
#                 closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#                 are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(face_locations))]
#
#                 # Draw rectangles and display names on the video frame
#                 for (top, right, bottom, left), are_match in zip(face_locations, are_matches):
#                     if are_match:
#                         name = knn_clf.predict([faces_encodings[are_matches.index(True)]])[0]
#                         print("인증된 얼굴 이름",name)
#                     else:
#                         name = "unknown"
#
#                     # Draw a rectangle around the face
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#                     # Display the name below the face
#                     cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
#
#                 # Display the resulting frame
#                 cv2.imshow('Video', frame)
#
#                 # Break the loop when 'q' key is pressed
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#     #frame이 한 화면이다 사용자에게 전달해서 본인이 맞는지 확인한다면 좋을 것
#     #name 아이디가 될 것 자료가 없는 얼굴이라면 unknown
#     # Release the video capture object and close all OpenCV windows
#     cv2.destroyAllWindows()
#     cap.release()
#
#     return jsonify({'user_id':name,'frame':frame})


@app.route('/api/data3', methods=['POST', 'GET'])
@cross_origin()
def get_data3(): #얼굴인식 학습코드
    data = request.get_json()  # 유저 아이디를 받아오면 그 이름으로 train 폴더를 만듭니다.

    # 프로젝트 경로에 맞게 사용자의 얼굴 사진을 저장할 폴더 경로를 설정합니다.
    project_path = 'C:/Users/jey92/project/team/3차/project3/backend/templates/knn_examples'
    user_photos_path = os.path.join(project_path, 'train')

    # user_photos_path 폴더 안에 사용자의 아이디 이름의 폴더에 사진이 저장되어야 합니다.
    user_id_folder_path = os.path.join(user_photos_path, data['id'])
    os.makedirs(user_id_folder_path, exist_ok=True)

    # face_recognition_knn.train 함수를 사용하여 사용자의 얼굴 사진을 학습합니다.
    classifier = face_recognition_knn.train(user_photos_path,
                                            model_save_path="trained_knn_model.clf", n_neighbors=2)

    return jsonify({'message': 'User data received and trained successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
