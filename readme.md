Result 파일 경로
1. ringfit_web/upload/user.mp4  
업로드 된 유저 영상 위치, 이름(이름 변경할 수 있음)

2. 결과는 ringfit_web/static/result  
<이미지, 비디오>
1) result/images/data1.png
무릎 각도와 관련된 이미지

2) result/images/data2.png
허리 각도와 관련된 이미지

5) result/images/graph.png
속도 관련 그래프(유사도 관련된 그래프)

<json>
{"1": "무릎각도차이", "2":"허리각도차이", "3":"앉는속도차이", "4":"일어나는속도차이", "5":"유사도"}

result/json/result.json
으로 주시오

main으로 띄울 영상(스켈레톤 입힌거): result/static/videos/result.avi
