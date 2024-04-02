import React, { useEffect, useRef, useState } from 'react';
import './loading.css';  // 위에서 작성한 CSS 파일 import

const WebcamStream = () => {
  const [showImage, setShowImage] = useState(false);
  const [loading, setLoading] = useState(false);
  const [id, setId] = useState('');

  const recognition = async () => {
    try {
      setLoading(true);

      // 캠 스트림을 가져옵니다.
      const response = await fetch('http://localhost:5000/api/video_feed3');
      console.log("리스폰 확인", response);

      // 응답에서 id를 추출합니다.
      const result = await response.text();
      const nameMatch = result.match(/data: (.+)\r\n\r\n/);

      if (nameMatch) {
        const recognizedId = nameMatch[1];
        console.log("인증된 id:", recognizedId);

        // 이미지 표시
        setShowImage(true);
        setId(recognizedId);
        // 0.1초의 지연 도입
        await new Promise(resolve => setTimeout(resolve, 10000));
        console.log('0.1초후 실행 ', recognizedId);
        // 컨트롤러에서 인식된 id로 데이터를 가져옵니다.
        window.location.href = 'http://localhost:8080/loginWebcam?id=' + recognizedId;
      }
    } catch (error) {
      console.error('웹캠 불러오기 에러', error);
    } finally {
      setLoading(false);
    }
  };

  const handleButtonClick = async () => {
    await recognition();
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '50vh',
      marginBottom: '280px',
    }}>
    {loading && !showImage && (
      <>
      <div >
        <div className="spinner" style={{ marginBottom: '10px' }} />
      </div>
      <p style={{ marginBottom: '20px' }}>얼굴 인식중...</p>
      </>
    )}

      {showImage && (
        <div style={{ marginTop: '270px' }}>
          <img
            src="http://localhost:5000/api/video_feed3"
            alt="Video"
            onLoad={() => setLoading(false)}  // 이미지가 로드되면 로딩 상태를 false로 설정
          />
          <p style={{ alignItems: 'center' }}>인증 된 아이디 : {id}</p>
        </div>
      )}

      <button onClick={handleButtonClick} style={{
        fontSize: '40px',
        padding: '5px 5px',
        border: '4px solid pink',
        background: 'white',
        marginBottom: '20px',
      }}>📸</button>

      <p>웹캠으로 얼굴 인증을 원하시면 카메라를 클릭 해주세요!</p>
    </div>
  );
};

export default WebcamStream;
