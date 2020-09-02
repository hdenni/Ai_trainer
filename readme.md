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


docker run -d -it --rm --name ringfit --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 cwaffles/openpose
/*
-d는 백그라운드에서 실행
-it -rm은 컨테이너를 닫으면 자동으로 컨테이너 삭제하는 옵션
--name ringfit는 컨테이너 이름을 ringfit로 해줌 안쓰면 지맘대로 랜덤으로 정해짐
--runtime 이후는 그래픽카드 쓰는 옵션인데 여러개 글카중에 1번 글카 쓴다는 옵션임
cwaffles/openpose는 도커 이미지 이름
*/

docker attach ringfit
// docker 여는거

./build/examples/openpose.bin --video examples/media/input.mp4 --write_json output/ --write_video path/~~.avi --display 0 --net_resolution "1312x736"--scale_number 8 --scale_gap 0.125

// scale_number 8배로 frame 늘림(정확도를 높이기 위함)
// write 형식은 반드시 avi
