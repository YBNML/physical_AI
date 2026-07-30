# Galbot G1 플랫폼 조사 — 원시 facts 국문 요약본

*(국문 요약본. 영문 원문 전문은 [11-galbot-g1-platform-facts-en.md](11-galbot-g1-platform-facts-en.md). 이 facts의 종합과 직접 검증(URDF 파싱·MuJoCo 실행)은 10-REVISION-02에 반영되어 있습니다.)*

출처 등급: **공식**(official — 벤더 문서·코드) / **논문**(paper) / **서드파티**(third-party media) / **직접측정**(own measurement).

---

## 토픽 1 — 기구학: head-frame 포즈 인터페이스 + 14-D 팔 관절 저수준 모델이 잘 정의되는가

### VERIFIED

| 사실 | 수치 | 출처 |
|---|---|---|
| 공식 DoF 구성(섀시·EE 제외): 목 2, 팔 7×2, 허리 3, 다리 2 | 합계 21 관절 (일부 AI 요약의 23/28은 오합산) | 공식 스펙 (developer.galbot.com G1 v2.2.4; 미러 robotsj.cn; robozaps 리뷰 교차확인) |
| DECISIVE: 공식 데이터 파이프라인은 23-D 관절 벡터를 하드코딩 | leg(0-4)=5, head(5-6)=2, left_arm(7-13)=7, left_gripper(14)=1, right_arm(15-21)=7, right_gripper(22)=1; robot_type "G1_V2.2B" | 공식 코드 (galbot-mcap2lerobot joint_constants.py) |
| DECISIVE: head·leg/waist 관절이 ACTION 벡터에 포함 — 시연 중 head가 명령됨(고정 장치 아님) | action[t]=state[t+1], 23-D 전체; 팔은 14/23뿐 | 공식 코드 (galbot_mcap_to_lerobot.py L306-307, 633, 663-667) |
| 헤드 카메라는 2-DoF 목 위 스테레오 페어; 양팔에 손목 카메라 | 헤드 스테레오 2×(480×640)@30fps; 토르소 RGB 4; 손목 depth 2; 손목 6축 F/T 2; 3D LiDAR + 초음파 8; head-left가 데이터셋 정렬 기준 | 공식 스펙+코드 (config.py IMAGE_TOPICS) |
| 손목 6축 F/T 2× 하드웨어 존재하나 공식 데이터 파이프라인은 미출력 | 하드웨어 2개 / LeRobot 출력 F/T 채널 0 (use_effort는 모터 토크일 뿐, 손목 F/T 아님) | 공식 스펙+코드 |
| 팔 7-DoF×2 확정 — 3개 소스 클래스 독립 확인 | 팔당 7, 총 14 | 공식 스펙·코드; arXiv:2602.12215, arXiv:2501.04595 |
| 팔 기하·페이로드 | 어깨-손목 710mm; 손목 작업공간 수직 0-2100mm / 수평 1900mm; 페이로드 5kg/팔, 합 10kg; 마케팅 스팬 1.9m·'0-2.4m'는 홍보용 envelope | 공식 스펙 |
| 4륜 옴니 베이스, 평면 3 DoF; 23-D 벡터에 base 포즈·오도메트리 부재 | 600×600mm, 투영원 625mm, 최대 1.5m/s | 공식 스펙 (DoF 수는 'omnidirectional'에서 추론) |
| 토르소 높이는 leg/waist 그룹으로 650mm 가변 ('folding legs') | 650mm 승강; leg+waist 5관절 (다리2 + 허리3) | 공식 스펙; worldrobotconference.com, baike.baidu.com |
| EE 교체형, G1 V2.2B는 측당 1-DoF 그리퍼, aperture는 연속값 | raw/1000 = 폭(m), 이진 개폐 아님; Isaac Lab charlie 변형은 좌 평행 그리퍼+우 석션; 옵션 22-DoF Sharpa 핸드 | 공식 코드; NVIDIA Isaac Lab galbot.py; arXiv:2602.12215 |
| NVIDIA Isaac Lab 공식 Galbot 태스크는 모든 조작 포즈를 BASE frame으로 표현 | 관측 항목 전부 *_in_base_frame, FrameTransformer 루트 base_link | Isaac Lab (stack_joint_pos_env_cfg.py, PR #3210) |
| 물리 외형·질량·전원·컴퓨트 | 1730mm; 92.5kg; 48V 30Ah; 8h 구동/3.5h 충전; 0-40°C; Jetson AGX Orin 64GB 275 TOPS ('G1 Premium'은 Thor 보도); 5.5in 1920×1080; WiFi 2.4/5GHz, BT 5.2 | 공식 스펙 |
| 서드파티/리셀러 스펙 페이지는 공식 매뉴얼과 상충 — 신뢰 불가 | 주장 47 DoF / 12-DoF 핸드 / 85-88kg vs 공식 21관절 / 92.5kg | 서드파티 (robotsinternational.com, canadasatellite.ca, humanoid.guide, aixzd.com) |
| 가격(대략) | 플래시세일 ~63만 RMB(~US$91k, 2026-02); JD.com 상시 69-70만 RMB | 서드파티 (blog.robozaps.com/b/galbot-g1-review) |

주요 출처: http://developer.galbot.com/docs/g1/2.2.4/zh/g1/ · https://www.robotsj.cn/shiyongshouce/1654.html · https://blog.robozaps.com/b/galbot-g1-review · https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot · https://github.com/isaac-sim/IsaacLab (PR #3210)

### UNVERIFIED (찾아봤으나 확인 실패 — 벤더 문의 목록)

- **정확한 링크 트리 / 어깨가 목·허리 3관절 대비 어디에 장착되는지.** 공개 URDF·MJCF·TF 트리 문서 없음(galbot_s1_description과 galbot_one_golf_description은 있으나 galbot_g1_description은 없음; Isaac Lab charlie.urdf는 Nucleus 서버 전용이고 leg 4관절로 G1과 다른 모델). head→arm-base 결론은 "허리 3 DoF가 전부 어깨 마운트 아래, 목과 양어깨가 한 강체 토르소 링크에 부착"이라는 표준 휴머노이드 가정에 의존. *(참고: 토픽 3에서 공식 galbot_one_golf_description(URDF/MJCF/USD, Apache-2.0)이 확인됨. 이후 REV.2에서 URDF 직접 파싱으로 구조 확정)*
- **'leg' 5관절이 PRISMATIC(텔레스코핑 컬럼)인지 REVOLUTE(폴딩 레그)인지.** 매뉴얼 '승강 높이차 650mm', 제품 카피 '折叠腿/folding legs', 리뷰 'lifting column' — 서로 함의가 다름. SDK 'leg' 5관절의 다리(2)/허리(3) 분할과 축 방향 미확인.
- **목 관절 가동 범위.** pan/tilt 각도 한계가 어느 소스에도 없음. *(참고: 토픽 3의 URDF에서 head_joint1 ±1.5208 rad, head_joint2 -0.2143..0.4936 rad 확인. REV.2: 틸트 총 40.6°뿐이라 조작 중 토르소 상시 이동 필요로 판정)*
- **헤드가 자동 서보(active vision/시선 추종)인지 텔레옵인지.** 녹화에서 명령되는 것은 확인했으나 어떤 정책에 의한 것인지 불명.
- **조작 중 목을 잠글 수 있는지 / SDK에 'freeze head' 모드가 있는지.** 공개 문서에 없음.
- **팔 반복정밀도.** 하드웨어 수치 부재. 널리 인용되는 '6자유도 조작 정밀도 오차 0.5mm 미만'은 Open6DOR **시뮬레이션** 벤치마크(2,500 태스크) 결과이지 기계적 반복정밀도 스펙이 아님 — 0.5mm를 반복정밀도로 취급하지 말 것.
- **팔 관절 속도·가속도·토크 한계.** 전 소스에서 부재. 공개된 유일한 속도 수치는 섀시 최대 1.5m/s.
- **팔 제어 주파수 / 관절 명령 인터페이스 서보 rate.** 데이터셋 fps 30(카메라 제한)이나 기저 컨트롤러 rate 미기재.
- **헤드 카메라 intrinsics, 스테레오 baseline, FOV, 센서 모델.** 해상도(480×640)와 双目(스테레오)만 확인. 헤드가 depth를 생성하는지 불명 — MobileH2R는 'head depth camera'라 부르나 매뉴얼은 双目相机로 상충.
- **SDK가 head-camera→arm-base 변환이 포함된 TF 트리를 퍼블리시하는지.** SDK 문서 페이지에서 클래스명·좌표계 정의·TF 토픽을 확인 못 함. 23-D 관절 피드가 있으므로 URDF만 있으면 계산은 가능.
- **G1 V2.2B 기본 EE와 좌우 구성이 소스 간 상충.** 중국 제품 카피: 좌 석션+우 적응형 그리퍼; Isaac Lab charlie: 좌 평행 그리퍼+우 석션; G1_V2.2B 관절 벡터: 양측 'gripper' 1-DoF. 동일 구성일 수 없음.
- **22-DoF SharpaWave 핸드가 공식 지원 옵션인지 연구 일회성인지.** 근거는 LDA-1B 논문뿐; Galbot 자료·관절 벡터는 1-DoF 그리퍼만.
- **그리퍼가 force 제어 가능한지 위치 전용인지.** 벡터는 폭만 저장; force/전류 명령 채널 미문서. *(참고: 토픽 2에서 그리퍼는 velocity·effort 명령이 유효한 유일한 관절군으로 확인)*
- **베이스의 holonomy 형식 확인.** 'omnidirectional'과 '4륜'만 명시, 휠 타입(mecanum vs 스티어드 swerve) 불명.
- **베이스 오도메트리가 존재하고 관절 스트림과 동기화되는지.** 공식 컨버터는 오도메트리 토픽을 아예 기록하지 않음 — 기준 데이터셋 포맷에 base 운동이 전무.
- **베이스·핸드 포함 공식 총 DoF 진술.** 공식 표는 의도적으로 '섀시·EE 제외' 스코프; '47 DoF'는 서드파티 전용이고 공식 표와 정합 불가.

### IMPLICATIONS (핵심 압축)

- **직접 답**: head 카메라→arm base 체인은 front_head_camera → head_joint2(tilt) → head_joint1(pan) → 토르소 링크 → [고정 어깨 마운트]. 목 관절 **2개**만 경유하고 팔 관절은 0개. leg/waist 5관절과 평면 base 3 DoF는 양쪽 브랜치의 공통 조상이라 상대 변환에서 대수적으로 상쇄. 단, 허리 3관절이 어깨 아래라는 미검증 가정에 의존. *(참고: REV.2에서 URDF 파싱으로 head→arm_base가 목 관절 2개에만 의존함이 확정)*
- **정량화**: 14 팔 관절만으로는 head→arm-base 변환의 **2 스칼라(pan·tilt)가 구조적으로 결손**. 카메라가 목 피벗에서 lever arm 위에 있어 pan+tilt가 6-DoF 변환 중 약 5개 성분을 움직임. 해결은 저렴: Model 2 state 입력에 head_joint1/2 추가 (14→16).
- **"head는 고정" 가정은 벤더 코드로 반박됨** — head-image frame의 타깃은 움직이는 기준계의 타깃. 선택지: (a) 목 잠금(freeze 기능 미확인), (b) head 관절을 Model 2에 입력, (c) Model 1이 base_link frame으로 출력. NVIDIA 공식 태스크는 (c).
- 절대(world/테이블 고정) 타깃이 필요하면 비관측 수가 2→**10**(목2+leg/waist5+base3)으로 폭발. 650mm 토르소 승강이 sharp edge — reachability와 자기충돌 가능성을 바꿈.
- Model 2 출력 14-D는 벤더 시연 action 공간(23-D)을 재현 불가 — 그리퍼 2·목 2·leg/waist 5를 명령 못 함. 최소 16-D(팔14+그리퍼2), 그리퍼는 연속 폭 회귀(분류 아님).
- F/T는 하드웨어에 실재하나 기본 데이터 경로에 없음 — 컨버터를 직접 확장해야 함(설정 플래그가 아님).
- 7-DoF 팔은 redundant(7>6): 팔당 1-D null space → Model 2는 redundancy resolution을 학습하거나 제공받아야 함. 시연의 '자연스러운' null-space 자세는 토르소 높이와 상관되므로 leg 관절도 컨텍스트로 입력할 이유.
- 스테레오 헤드는 Model 1에 유리(추가 하드웨어 없이 metric depth). 컴퓨트는 탑재 Orin에서 Model 2 실행 고려 — 외부 Mac 경유 왕복 latency가 지배적 설계 리스크. *(참고: REV.2에서 추론은 탑재 AGX Orin으로, REV.3에서 학습·시뮬은 RTX 3090/Ubuntu 박스로 확정)*
- 권고 최소 수정: 인터페이스 확장 in 14→21(+head2, +leg/waist5), out 14→16(+그리퍼2), 컨버터에 F/T 추가, 또는 Model 1 출력 frame을 base_link로 변경. 실기체에서 3가지 확인: URDF 구조 / 목 잠금 여부 / 실제 EE 구성.

---

## 토픽 2 — 센서·제어 인터페이스 (2모델 대뇌/소뇌 아키텍처 대비 평가)

### VERIFIED

| 사실 | 수치 | 출처 |
|---|---|---|
| 공개 개발자 포털+공개 SDK 존재 — 판매상담 전용 로봇 아님 | SDK v1.9.1(2026-07-03), robot env GBS V1.17, Ubuntu 20-24, Python 3.8-3.14; C++ API ref ~877KB + 예제 ~60개, NDA 불요 | 공식 (github.com/GalaxyGeneralRobotics/GalbotSDK) |
| **DECISIVE: 팔은 관절 위치 명령 전용.** "For standard joints (head, legs, arms), only JointCommand::position is effective in current versions; velocity, acceleration, and effort are currently ignored." (표준 관절은 position만 유효, 나머지는 무시) | JointCommand 필드: position(rad)/velocity/acceleration/effort(N·m)/Kp/Kd — 팔·머리·다리는 position만 반영. impedance·가변 stiffness·Cartesian force 모드·Kp/Kd 튜닝 전부 없음 | 공식 C++ API ref |
| 손목 6축 F/T 실재·표준 — 독립 이중 확인 | 2개(손목당 1), 6축; force N / torque N·m / timestamp_ns; get_force_sensor_data(LEFT/RIGHT_WRIST_FORCE); 전용 예제 존재 | 공식 스펙 + SDK 예제 |
| 헤드는 스테레오 RGB 페어, 네이티브 depth 없음 | HEAD_LEFT/RIGHT_CAMERA 2개뿐; depth는 FOUNDATION_STEREO / LIGHT_STEREO 학습 파이프라인으로 계산 — 센서 read가 아닌 GPU 추론 비용 | 공식 SDK enum |
| 손목 카메라는 RealSense급 완전 RGB-D, 손목당 4스트림 | color+depth+IR×2 = 4/손목, 총 8; IrData 문서 "e.g., RealSense infra1/infra2" | 공식 API ref |
| 관절 피드백은 토크+모터 전류 포함 — 관측이 명령보다 훨씬 풍부한 비대칭 인터페이스 | JointState 7필드: position/velocity/acceleration/effort(N·m)/current(A) + ns 타임스탬프 | 공식 API ref |
| per-frame 모델 추론용 고주파 스트리밍 명령 경로가 명시적으로 설계됨 | set_joint_commands(무보간, 목표 즉시 추종) + set_joint_commands_batch("VLA inference batch output"용); set_joint_positions는 고주파 부적합 경고 | 공식 API ref |
| Cartesian EE 포즈 채널(WBC) 존재 — Model 1 출력에 거의 그대로 대응 | 포즈 7값 [x,y,z,qx,qy,qz,qw], 기본 reference 'world'; 판독 lee_pose/ree_pose/head_pose; stiffness/wrench 인자 없음 | 공식 API ref (set_end_effector_command / get_wbc_end_effector_poses) |
| 림별 전환식 PVT 컨트롤러 + 미문서화 'BYPASS' 변형 존재 | 명명 컨트롤러 14종, lifecycle 함수 6종; PVT 명명은 모터층이 Position-Velocity-Torque 지원임을 시사 | 공식 API ref + controller_management_example.cpp |
| 외부 호스트의 Ethernet 제어는 정식 'PC' 배포 모드 | 3노드: PC(192.168.1.99) / XCU(192.168.1.66) / HPU(192.168.1.88), 'embosa' 미들웨어; 스크립트에 SSH 자격증명 하드코딩(XCU root/<redacted>, HPU galbot/<redacted>) | 공식 SDK config (configure_embosa_ip.sh) |
| SDK는 Linux 전용 — macOS 미지원, M4 Mac mini 플랜의 hard blocker | Ubuntu 20-24만; x86 PC→ARM 로봇 크로스컴파일 워크플로; Darwin 언급 0 | 공식 README |
| ROS 네이티브 아님, ROS 2 메시지 형태 호환; 기록은 MCAP protobuf | ROS 2 타입 6+개 미러(CameraInfo 등); ROS 배포판 명시 없음 | 공식 API ref |
| 공식 텔레옵→모방학습 파이프라인, 30Hz 위치 타깃 | 23-D state/action; action[t]=state[t+1]; 4캠 480×640@30fps; head-left가 타이밍 앵커, scipy interp1d 정렬 | 공식 (galbot-mcap2lerobot) |
| 공식 텔레옵 기록에 손목 F/T·EE 포즈 없음 | parquet 컬럼 7개; 캡처된 F/T 채널 0 | 공식 |
| 표준 구성에 지첨 tactile 없음 | tactile enum 0; 석션컵 진공압(Pa)만; Sharpa 덱핸드 장착 시에만 22관절+force; 핸드 옵션 Inspire/BrainCo/Sharpa/Linker Hand L20. "hands에 tactile 통합" 영문 주장은 근거 없음 | 공식 SDK enum |
| 공식 하드웨어 스펙 총표(서드파티 수치 대체) | 1730mm/92.5kg/710mm/650mm/0-2100·1900mm/5·10kg/1.5m/s/48V 30Ah·3.5h·8h/0-40°C/터치스크린·스피커·마이크/USB3.0×2+HDMI | 공식 |
| 탑재 컴퓨트 AGX Orin 64GB + 별도 노드 최소 2개 | 275 TOPS; XCU(실시간 제어 추정)+HPU(Orin 추정) 모두 SSH 접근 가능 | 공식 스펙 + SDK |
| SDK enum 기준 전체 센서 로스터 — "LiDAR 없음" 주장 반박 | 팔/헤드 카메라 10 + 서라운드 컬러 4 + 3D LiDAR 1 + 초음파 8 + IMU 3; init()에서 센서 화이트리스트 필수(연산·메모리 절감) | 공식 SDK enum |
| 시각 동기화 멀티카메라+관절 관측 프리미티브 내장 | get_synced_observation: 첫 카메라가 anchor 클록, 최근접 정렬, delta_to_anchor_ms 보고; enable_sync_mode 플래그 | 공식 |
| 모션플래닝·IK·FK·Jacobian 노출 — 단 충돌 월드는 인지 비연동 | "does not have real-time obstacle perception"; add_obstacle() 수동 등록; 충돌 이산화 0.01m | 공식 GalbotMotion ref |
| e-stop 있음, 안전 인증 근거는 0 | ControlStatus fault 코드 10종; ISO 10218 / 13849 PL / TS 15066 / TUV / CE 언급 전무 | 공식 |
| SDK 자체 '실시간 제어 루프' 예제가 실시간이 아님 — 달성 가능 rate에 대한 red flag | example8: waypoint당 1s sleep, max_speed 0.1rad/s; SDK 전체에 고주파 폐루프 예제 0, 루프 주파수 명시 0 | 공식 예제 소스 |
| 그리퍼만 force/velocity 명령이 유효한 유일한 곳 | "position field represents gripper width and both velocity and effort fields are supported"; 기본 effort 파라미터 5; 덱핸드 범위는 벤더별 | 공식 API ref |
| 서드파티 영문 소스는 신뢰 불가·상호 모순 | 85kg/47DoF/10h/노LiDAR/tactile/95-97%/$87K 주장 다수가 공식과 상충; 기본 EE '좌 석션+우 그리퍼' 보도는 SDK enum과는 정합 | 서드파티 |

주요 출처: https://github.com/GalaxyGeneralRobotics/GalbotSDK · https://developer.galbot.com/docs/g1/2.2.4/zh/g1 · https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot

### UNVERIFIED (찾아봤으나 확인 실패 — 벤더 문의 목록)

- **제어 루프 rate — 단일 최대 공백.** 274K자 API 레퍼런스 전체에서 'Hz/frequency/rate/latency' 검색 — 'high-frequency streaming'을 반복하나 숫자를 단 한 번도 명시하지 않음. set_joint_commands 최대 명령 rate, WBC 내부 rate, 최소 명령 간격 전부 미문서. 실측하거나 Galbot에 직접 물어야 함.
- **관절 상태/센서 퍼블리시 rate.** get_joint_states·F/T·카메라 모두 rate 미기재. 30 FPS는 변환 타깃이지 소스 rate 보장이 아님.
- **실시간 보장.** 결정성·jitter 한계·RT 커널·XCU의 RTOS 여부 전무. 'embosa'의 QoS·전송(TCP/UDP) 미공개.
- **외부 PC→팔 동작 end-to-end latency.** PC→XCU/HPU 홉 비용 vs HPU 직접 실행 차이 불명.
- **헤드 카메라 모델·네이티브 해상도·프레임률·FOV.** 480×640은 텔레옵 기록에만 등장(다운스케일일 수 있음). FOV·벤더 미상.
- **손목 카메라 정확한 모델.** 'e.g., RealSense infra1/infra2'로 강하게 시사되나 모델 번호(D405/D435 등)·해상도·depth 범위 미공개. *(참고: 토픽 3의 xacro에서 D405(1280×720, d415 옵션)로 확인)*
- **F/T 센서 제조사·모델·성능.** 측정 범위(N/N·m)·분해능·노이즈 플로어·샘플링 rate·과부하 한계·중력/페이로드 보상 여부·wrench 장착 frame 전부 미공개.
- **손목 F/T가 전 G1 유닛 표준인지.** enum명 GalbotOneFoxtrotSensor는 특정 변형('Galbot One Foxtrot')을 시사; 데이터 repo는 'G1_V2.2B'. 스펙 표에 '옵션' 표기가 없는 것이 표준이라는 최강 근거이나, 구매 유닛에 대해 벤더 확인 필요.
- **PVT vs PVT_BYPASS 의미.** 'bypass'가 무엇을 우회하는지(안전 리미터? WBC? 보간기?), 토크/속도 명령을 여는지 문서 0. sub-position 제어를 원하는 이에게 가장 흥미로운 미문서 단서.
- **effort/velocity 명령이 활성화될지.** 'in current versions'는 변경 가능성을 시사하나 로드맵·타임라인 없음.
- **안전 인증.** ISO 10218·13849 PL·TS 15066·TUV·CE 어디에도 없음. 인증된 안전 컨트롤러의 존재 자체가 불명.
- **안전 레이어 우회 가능 여부.** '_BYPASS_CTRL'과 root SSH 접근이 깊은 접근을 시사하나 결과는 전적으로 미문서.
- **내부 모터 버스.** XCU↔관절 드라이브 간 EtherCAT/CANopen 등 언급 없음. 외부 IP 레이어만 문서화.
- **관절별 최대 속도/가속/토크 한계.** SDK 기본값(0.1-0.2 rad/s)은 보수적 기본값이지 하드웨어 한계가 아님.
- **팔 관절 기구학.** 공개 DH 파라미터·관절 한계 없음; galbot_g1_description repo 부재. *(참고: 토픽 3의 galbot_one_golf_description URDF에 전 관절 한계·effort·속도 공개 — REV.2에서 파싱 확정)*
- **공식 시뮬레이션 지원.** G1 전용 Isaac Sim/MuJoCo/Gazebo 모델·sim-to-real 툴킷을 못 찾음. *(참고: 토픽 3에서 공식 MJCF/URDF/USD 확인, M4에서 MuJoCo 구동 검증됨)*
- **macOS 지원.** SDK가 명시적으로 'macOS 미지원'이라 쓰지 않음 — 전면적 부재로부터의 추론(사실상 확실). *(참고: 이후 REV.2에서 Linux 전용으로 확정 — Mac은 로봇과 통신 불가)*
- **중국 외 개발자 유닛 가격·가용성·리드타임.** $87K는 서드파티 미확증.
- **set_end_effector_command가 stiffness/wrench를 받는지.** 시그니처는 포즈·frame뿐. WBC 내부 임피던스 거동·튜닝 가능성 미문서.
- **SingoriXTarget 내용.** 최저수준 PublishTarget/RequestTarget 경로의 멤버 필드 미열거 — 가장 깊은 노출 제어 채널이며 set_joint_commands 이상을 허용할 수도 있음.
- **텔레옵 하드웨어.** Galbot 자체 텔레옵 rig(VR/외골격/leader arm)·판매 여부·기록 rate 미공개. *(참고: 토픽 3에서 TM01 leader-follower rig 확인)*

### IMPLICATIONS (핵심 압축)

- **Model 2는 생존하나 '소뇌'로서는 아님.** position-only가 결정적 — 토크·stiffness·damping 변조 불가. 컴플라이언스/임피던스라는 생물학적 은유가 함의하는 모든 것이 이 인터페이스에서 불가. Model 2를 정직하게 재규정: 학습된 redundancy-resolver + 궤적 셰이퍼이지 force 컨트롤러가 아님.
- **"IK가 못 하는 것"의 답은 좁아졌지만 실재함**: 명령 채널이 IK와 동일하므로 Model 2의 가치는 전적으로 *더 나은 관절 타깃 선택*(null space 활용, 양팔 협조, 장애물 인지 자세, 사람 텔레옵 스타일)에서 나옴. GalbotMotion이 실시간 장애물 인지가 없다는 점이 구체적 공백 — vision 조건부 정책의 방어 가능한 논지. '학습된 컴플라이언스'는 아님.
- **F/T 의존성 성립(이중 확인).** 단 F/T는 관측으로만 진입해 위치 타깃을 이동시킴 → Model 2는 사실상 admittance 컨트롤러이고, admittance 품질은 루프 rate에 전적으로 종속 — 그 rate가 미문서. **정책 코드 작성 전 제어 rate 실측이 최우선·최저비용 실험.** SDK 자체 예제가 1Hz blocking 데모라는 것이 진짜 경고 신호.
- **Mac mini 플랜은 서술된 대로면 blocked.** 선택지: (a) 탑재 Orin에서 추론(네트워크 홉 제거, 275 TOPS/64GB가 이 워크로드엔 Mac보다 강함), (b) 로봇 LAN에 소형 Linux 박스를 SDK 호스트로, (c) Mac은 학습 전용. *(참고: REV.2/REV.3에서 (a)+학습용 RTX 3090 Ubuntu 박스 조합으로 확정)*
- **Model 1 인터페이스는 이미 만들어져 있음** — set_end_effector_command가 정확히 양손 7-D 포즈를 받음. Model 1→WBC 직결이 강력한 baseline: 이것으로 충분하면 Model 2는 그 대비로 자신을 정당화해야 함. 이 실험을 일찍 돌릴 것.
- **head depth 결정 필요**: 헤드는 depth 센서가 없고 학습 스테레오는 Orin GPU를 정책과 경쟁시킴. 대안: 스테레오 두 장을 그대로 입력해 implicit depth 학습, 또는 근거리엔 실제 하드웨어 depth인 손목 RGB-D 활용.
- **수집 전에 파이프라인 확장.** F/T 소급 추가는 불가능 — 기록·변환에 wrench 채널을 먼저 추가. 벤더 action 규약(action[t]=state[t+1], 30fps 절대 관절 위치)은 플랫폼의 사실상 동작점이자 합리적 기본값.
- **30Hz 위치 타깃이 현실적 설계점 — 2모델 분리 근거를 약화.** 30Hz에서 '빠른' Model 2가 Model 1보다 유의미하게 빠를 여지가 작음. 명령 경로가 훨씬 빠르다는 증거를 실측으로 찾거나, set_joint_commands_batch 기반 단일 모델+action chunking 고려.
- 23-D state는 공짜 업그레이드(이미 데이터셋에 있음); JointState의 effort·전류는 손목 wrench를 보완하는 공짜 접촉 신호.
- **상업화 벽은 기술이 아니라 인증·보안.** 안전 인증 부재 + 하드코딩 root 자격증명(XCU root/<redacted>, HPU galbot/<redacted>)의 플랫 LAN — 둘 다 소프트웨어 레이어에서 못 고침. 지금 조달 질문으로 제기할 것.
- **벤더에 물을 최고가치 질문: LEFT_ARM_PVT_BYPASS_CTRL의 의미.** 토크/임피던스 플러밍은 스택에 명백히 존재하며 SDK가 'in current versions' 전달을 거부할 뿐. bypass나 SingoriXTarget이 토크·게인을 연다면 컴플라이언스 지향 Model 2가 전부 살아남.

---

## 토픽 3 — 시뮬레이션 자산·데이터셋·모델·텔레옵 (M4 Mac mini 시뮬-퍼스트 타당성)

### VERIFIED

| 사실 | 수치 | 출처 |
|---|---|---|
| 공식 Apache-2.0 로봇 기술 repo 존재: URDF+MJCF+USD+xacro — 'Galbot One Golf'가 곧 G1(렌더에 'G1' 각인) | 104 links / 103 joints(revolute 33, continuous 44, fixed 26); 640 files; 67MB; Apache-2.0 | 공식 (github.com/GalaxyGeneralRobotics/galbot_one_golf_description) |
| URDF는 물리 시뮬용으로 진짜 완전: inertia·관절/effort/속도 한계·collision 메시 전부 존재 | 91/104 links inertia; 78/104 collision STL; zero-inertia frame 링크 2; mimic 관절 10; **총질량 116.18kg vs 스펙 92.5kg**(차이 대부분 수동 옴니휠 롤러 40개 — float-base 동역학에 질량 예산 맹신 금지) | 직접 파싱 (urdf/galbot_one_golf.urdf) |
| 양팔 7-DoF 관절 한계·effort·속도 한계 공개 | (rad) j1 ±3.00432619; j2 ±1.608062789; j3 ±2.916972222; j4 비대칭 미러(우 -1.869862177..2.5679938779914946, 좌 반전); j5 ±2.916972222; j6 ~-0.735..0.823 미러; j7 ±1.538202778. effort(Nm) 60/60/30/30/10/10/10; 전 팔관절 1.5rad/s. leg effort 169/169/84/30/30; head1 pan ±1.5208, head2 tilt -0.2143..0.4936(4Nm/1.5rad/s); 그리퍼 0..1.703rad, effort 50 | 직접 파싱 |
| MJCF는 완전 구동형 시뮬 모델(시각화 export 아님) | 27 액추에이터(23 position + 4 velocity 휠, kp/kv/ctrlrange/forcerange); jointvel 센서 25; collision geom 184 + visual 83; 메시 138; timestep 0.002s; joint당 armature 0.1/damping 0.5/frictionloss 0.5; urdf-to-mjcf로 재생성 가능(토크 액추에이터 교체 가능) | 공식 (mjcf/galbot_one_golf.xml) |
| **직접 검증: 공식 G1 MJCF가 M4 Mac mini의 MuJoCo에서 무오류 로드·고속 구동** | mujoco 3.11.0; 3000 스텝 0.075s = 39,904 steps/s = **79.8× 실시간**(단일 코어); 오프스크린 렌더 720×1280 정상; nq=nv=33, nu=23; FK 정상(head_link2 기준 우 TCP [0.614,-0.259,-0.139]m) | 직접측정 (사용자 M4 Mac mini) |
| NVIDIA Isaac Lab에 first-party G1 지원 병합(upstream) | PR #3210(2025-09-09 merge); Gym 태스크 5종; isaaclab_mimic env 4종(데모 자동 확장); 후속 PR #3764(석션 관통 수정) — 활발히 유지보수. 자산은 CHARLIE 세대: 좌 평행 그리퍼+우 석션, leg 4관절 | 공식 (isaac-sim/IsaacLab PR #3210) |
| 손목 6축 F/T 2× 하드웨어 + 깔끔한 SDK API (재확인) | force(N)+torque(N·m)+ns 타임스탬프 | 공식 스펙 + SDK |
| SDK가 Model 1→Model 2 인터페이스를 그대로 제공: WBC EE 포즈 채널 + 관절 명령 채널 병존 | EE 포즈 7-D; lee/ree/head_pose; 그룹별 PVT(+BYPASS) 컨트롤러; publish_target 고주파 경로; 예제 궤적 dt=0.008s(125Hz); 문서가 스트리밍 API 사용을 명시 권고 | 공식 SDK |
| Galbot이 G1 전용 first-party 7-DoF leader-follower 텔레옵 rig 'TM01'을 판매 | 7-DoF×2 leader arm(G1 팔과 isomorphic); 샘플링 ≥100Hz; 매핑 latency ≤20ms; 530×330×70mm ~2kg; 12V/30W USB; 아날로그 트리거=연속 그리퍼; 에피소드 record/segment/mark-failed/discard 전용 버튼+음성 안내; 조이스틱으로 토르소·섀시·헤드; 손목 示教(티칭) 버튼으로 드래그 티칭 | 공식 (developer.galbot.com/docs/tm01/1.4.0/en/tm01) |
| 공식 MIT MCAP→LeRobot v2.1 컨버터가 실기체 텔레옵 데이터 스키마를 문서화 | 23-D state/action(leg→head→left_arm→l_grip→right_arm→r_grip); 그리퍼 raw/1000=m; 4캠 480×640@30fps(헤드 스테레오 양쪽 기록); head-left가 마스터 클록; action[t]=state[t+1]; 출력 Parquet+AV1; MIT | 공식 (galbot-mcap2lerobot) |
| 실기체 G1 양팔 에피소드 ~3,000개가 HuggingFace에서 지금 다운로드 가능 (BAAI RoboCOIN) | 판독 가능 5세트: fold_clothes 593/520,473 · fold_clothes_1 581/536,843 · use_dryer 952/450,316 · use_dryer_1 620/242,533 · steamer_baozi 228/270,556 = **2,974 ep / 2,020,721 frames / ~18.7h**; Galbot 세트 총 18; 카메라 3(헤드 480×640 + 손목 368×640×2); state 21-D, action 16-D(팔14+그리퍼2); **eef_sim_pose_state/action 12-D(양손 6-D 포즈) 동봉**; 30fps; 21-D는 waist 3만 기록되어 SDK 23-D와 정합 | 공식(BAAI) + HF API; arXiv:2511.17441 |
| 공식 스펙시트 v2.2.4 (요약) | 1730mm/710mm/650mm/0-2100·1900mm/5·10kg/92.5kg/0-40°C; Orin 64GB 275 TOPS; DoF(섀시·EE 제외) 목2+팔7×2+허리3+다리2 *(주: 원문 토픽 3이 '=19'로 오기 — 실합은 21, 토픽 1 참조)*; 48V 30Ah 8h; 옵션: 무선충전 독·예비 배터리·리모컨 | 공식 |
| 실구매 가능 — JD.com 상장, 2026-02 CCTV 춘완 갈라 후 하룻밤 ~300대 판매 | ¥630,000(~$87K); Galbot 자체 무인약국·편의점 운영 병행 | 서드파티 (finance.eastmoney.com/a/202602173651264736.html) |
| 개발자 지원 구독 티어 판매; 시뮬 자산은 L1 정식 상품 | FREE ¥0(SDK·문서·디지털 모델·커뮤니티) / PRO ¥5만/로봇/년(SLA P0≤4h, 원격 디버그, 온라인 시뮬+실기 배포) / ENTERPRISE ¥20만/로봇/년(전담 엔지니어·현장 지원 해외 포함·P0≤2h); '端到端模型动作指令直通'(end-to-end 모델 액션 명령 직통) 명시 | 공식 (developer.galbot.com) |
| GalbotSDK는 Linux 전용 (재확인) | v1.9.1; 바이너리 타깃 linux-x86_64/aarch64-gcc940만; 라이선스 NOASSERTION(커스텀, 비OSI); repo 34,073 파일; FoundationStereo·내비·덱핸드(INSPIRE/BRAINCO/SHARPA — SHARPA는 per-finger force 반환) 예제 포함 | 공식 |
| Galbot 플래그십 조작 모델 GraspVLA: 코드+웨이트 공개, 단 실기체는 Franka Panda(G1 아님), 라이선스 CC BY-NC(상업 불가) | frozen DINOv2+SigLIP → InternLM2-1.8B → flow-matching action expert; chunked EE delta action; SynGrasp-1B 순수 합성 사전학습(1e9 frames, 10,680 objects/240 cat); 추론 200ms/~9GB(L40s); 카메라는 외부 RealSense 2대 | 논문 arXiv:2505.03233 + github.com/PKU-EPIC/GraspVLA |
| CRITICAL: GraspVLA는 계층형이 아님 — 상/하위 분리 없는 단일 end-to-end | 'Progressive Action Generation'은 한 네트워크 내부 CoT(bbox→grasp pose→action chunk), 공동 학습 — 포즈 인터페이스의 2모델 분리가 아님. 벤더는 제안된 대뇌/소뇌 분리를 이 로봇에 발표한 적 없음 | 논문 |
| Galbot의 다른 공개 모델은 내비/추적, 양팔 조작 아님 | TrackVLA(1.7M 샘플, 10FPS)·TrackVLA++; org 대형 repo는 locomotion 계열; MobileH2R(CVPR 2025)는 G1을 실플랫폼으로 사용, 순수 합성 학습, baseline 대비 +15% | arXiv:2505.23189, 2510.07134, 2501.04595 |
| TRAP: Galbot의 OpenWBT 텔레옵 repo는 **Unitree** G1/H1용 | Apple Vision Pro 전신 텔레옵; Galbot G1 아님 — 'G1 teleoperation' 검색 시 필터링 필수 | 공식 README (OpenWBT) |
| RoboTwin 2.0은 Galbot G1 미지원 | 지원 embodiment 5종: Aloha-AgileX, Franka, UR5, Piper, ARX-X5; 신규 embodiment 추가 경로는 문서화(직접 작업 필요); SAPIEN 기반=Linux+NVIDIA *(참고: REV.3에서 시뮬 embodiment는 G1 대신 RoboTwin 2.0을 사용하기로 결정)* | 공식 문서 + arXiv:2506.18088 |
| macOS 시뮬레이터별 판정 | MuJoCo: Apple Silicon 완전 지원(직접 79.8× 검증, MJX도 Metal 가능); Isaac Sim/Lab: 불가(x86_64+NVIDIA RTX 전용); ManiSkill/SAPIEN: 'no support for MacOS'(CPU만); RoboTwin: 동일 제약; Genesis: Linux/macOS/Windows 문서화(MPS 경로) | 직접시험 + 각 공식 문서 |
| 커뮤니티 Isaac-Sim용 USD 자산 존재 — 세대 계보 노출 | Charlie → Foxtrot → Golf; SDK enum은 'GalbotOneFoxtrotSensor', 컨버터는 'G1_V2.2B'; fiveages-sim 2 repo(154/290 files, Apache-2.0) | 서드파티 (github.com/fiveages-sim/galbot-usds, /robot-descriptions-galbot) |
| 비대칭 EE 구성은 first-class 제품 변수 | robot.xacro에 left_ee_type/right_ee_type 독립 인자; 옵션: Galbot 평행 그리퍼/석션/hitbot 그리퍼/덱핸드 3종; Isaac Lab charlie=좌 그리퍼+우 석션; 중국 매체=좌 석션+우 그리퍼; 오픈소스 기본 프리셋은 양측 그리퍼 | 공식 xacro + Isaac Lab + SDK |
| SDK 레퍼런스 VLA 예제는 125Hz 관절공간 궤적 스트리밍 | example6: dt=0.008s(125Hz), 200포인트; 그룹별(leg5/head2/arm7+7) **절대 관절 위치** — EE delta 아님; 배포 타깃은 탑재 Orin | 공식 예제 + 스펙 |

주요 출처: https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description · https://github.com/isaac-sim/IsaacLab/pull/3210 · https://developer.galbot.com/docs/tm01/1.4.0/en/tm01 · https://huggingface.co (RoboCOIN org) · https://github.com/PKU-EPIC/GraspVLA · robotwin-platform.github.io

### UNVERIFIED (찾아봤으나 확인 실패 — 벤더 문의 목록)

- **오픈소스 URDF/MJCF/xacro에 헤드 카메라가 없음.** repo 전체 grep: head_end_effector_mount_link(빈 frame)만 있고 카메라 링크·optical frame·intrinsics·스테레오 baseline 전무. MJCF에 `<camera>`·`<site>` 요소 0. Model 1·2가 head-image frame으로 정의된 이상 **head-camera↔head_link2 extrinsic이 전체 아키텍처에서 가장 하중이 큰 수치인데 미공개** — 실기체 hand-eye 캘리브레이션으로 실측하거나 Galbot에서 받아야 함(SDK의 get_camera_intrinsic()은 런타임 로봇에서만 반환).
- **시뮬 자산에 F/T 센서 frame 없음.** xacro/urdf/mjcf/usd/config에서 force|torque|ft_sensor grep 0건. MuJoCo `<site>`+`<sensor>`를 직접 추가하고 장착 frame을 추정해야 함 — Model 2의 F/T 입력은 out of the box로 시뮬 불가.
- **헤드 스테레오 카메라 모델·baseline·해상도·FOV·프레임률.** 손목은 xacro에서 Intel RealSense D405(d415 옵션)로 식별되나 헤드는 부품 번호 미확인.
- **G1 실제 저수준 제어 루프 rate.** 예제는 125Hz 궤적 포인트를 쓰고 publish_target을 '고주파 경로'라 하나, WBC/관절 서보 rate를 명시한 문서 없음.
- **기록 텔레옵 데이터셋에 손목 F/T가 담기는지.** 공식 컨버터는 23-D 관절 위치만 출력(소스 protobuf의 velocity·effort도 폐기), F/T 토픽은 목록에 없음. RoboCOIN 공개 세트에도 F/T 채널 없음 — **Model 2가 요구하는 손목 F/T가 담긴 공개 G1 데이터셋은 현재 0개.**
- **'GraspVID'.** 영·중 반복 검색에도 해당 명칭의 Galbot 모델·논문·repo 없음. 1차 소스 확보 전까지 존재하지 않는 것으로 취급.
- **Galbot의 G1용 계층형/2단계 대뇌-소뇌 아키텍처 발표.** 없음. GraspVLA는 명시적 unified end-to-end; TrackVLA는 공유 backbone+2 head. 제안 아키텍처는 이 로봇에 벤더 prior art가 없는 자체 설계.
- **SynGrasp-1B 미공개.** 'Coming Soon' 상태 지속(코드·웨이트만 공개).
- **Isaac Lab Galbot USD의 단독 다운로드 가능 여부.** Nucleus 서버 의존; 내용·DoF·센서 탑재 미검증. Isaac Sim 공식 'Robot Assets' 표에는 Galbot 미등재('Galbot/galbot/g1.usd' 경로 주장은 확인 실패).
- **LeRobot upstream에 네이티브 Galbot G1 클래스 없음.** 통합은 외부 MIT 컨버터의 단방향뿐; lerobot 네이티브 record/replay/control 루프 부재.
- **Galbot 자체의 HF G1 데이터 없음.** 공개 G1 데이터셋은 전부 서드파티(BAAI RoboCOIN).
- **TM01의 단품 구매 가능 여부·가격·구독 없이 동작하는지.** 매뉴얼은 완비돼 있으나 가격·주문 경로 없음; 스펙시트의 '리모컨' 옵션과 TM01의 동일성 불명.
- **G1 주문 리드타임·중국 외 구매 지원.** ENTERPRISE의 '국내외 출장 가능'이 국제 역량을 시사하나 수출 조건·비중국 유통사 미확인.
- **신규 구매자가 받는 세대(Charlie/Foxtrot/Golf).** 오픈소스=golf(leg 5), Isaac Lab=charlie(leg 4), SDK enum=Foxtrot, 컨버터=G1_V2.2B — 상호 매핑 불명, 관절 수도 다름. 구매 시 세대와 일치 자산을 확인해야 함.
- **'12-DoF 핸드'·'47 총 DoF' 유포 수치.** 공식과 불일치; 애그리게이터 주장은 신뢰 불가.
- **URDF 팔 속도 한계 일률 1.5 rad/s(leg 1.0)는 placeholder로 보임.** 하드웨어 데이터시트와 교차검증 불가.
- **developer.galbot.com 온라인 시뮬(在线仿真)의 macOS·무구독 사용 가능성.** 백엔드 시뮬레이터·G1 카메라 노출 여부 불명(로그인 뒤).
- **Genesis의 G1 지원.** Genesis 일반론으로 Apple Silicon 지원 주장은 있으나 G1 자산·예제·통합 없음, MJCF/URDF 정상 인제스트 미시험.
- **Isaac Lab Galbot 태스크 5종+mimic env의 end-to-end 실행 여부.** upstream 코드·태스크 등록은 검증했으나 NVIDIA GPU 부재로 실행 못 함.
- **실기체 텔레옵→정책 latency 예산.** SDK 왕복 명령 latency·카메라 파이프라인 latency·WBC 추종 대역폭 수치 전무.

### IMPLICATIONS (핵심 압축)

- **시뮬-퍼스트 게이트 통과 — 구체적으로 MuJoCo로.** 공식 Apache-2.0 자산이 실제 inertia·한계·collision·튜닝 액추에이터를 갖췄고 이 M4에서 ~80× 실시간 구동을 직접 검증. 벤더 접촉 0으로 오늘부터 진짜 G1 기구학 대상 개발 가능.
- **단 Mac은 NVIDIA 생태계 절반에서 배제** — Isaac Lab first-party 태스크·Mimic 데이터 증폭·SAPIEN/RoboTwin 전부 Linux+NVIDIA 필요. Mac mini는 MuJoCo 개발 박스로, Isaac급 작업엔 별도 Linux+NVIDIA를 예산. *(참고: REV.3에서 RTX 3090/64GB/Ubuntu 박스 확보로 반영됨)*
- **Model 1/Model 2 인터페이스는 벤더 API를 계약으로**: Model 1 출력=양손 7-D 포즈(set_end_effector_command), Model 2 출력=14 관절 — 둘 다 출하 API에 직결. 무료 fallback: Model 2 자리에 벤더 WBC/RMPFlow를 꽂아 Model 1 단독 검증.
- **frame 선택 변경 또는 디커플.** 어떤 데이터셋·API도 head-image frame 포즈를 주지 않고, head 카메라 extrinsic은 미공개·미시뮬. 내부 표현은 유지하더라도 저장·교환은 공개 frame으로 하고, 헤드 스테레오 hand-eye 캘리브레이션을 명시적 초기 blocking 작업으로.
- **F/T는 하드웨어에 실재하나 시뮬과 전체 공개 데이터에서 부재** — MuJoCo `<site>`+`<sensor>` 직접 추가와 기록 파이프라인 확장을 first-class 작업으로 하거나, F/T 마스킹 시 graceful degradation 하도록 Model 2 설계. per-joint effort(protobuf·JointState에 존재)가 저렴한 부분 대체물.
- **로봇 구매 전 사전학습 가능**: RoboCOIN이 (양손 현재 포즈→목표 포즈) 감독쌍(Model 1)과 (목표 포즈+state→관절 action) 쌍(Model 2)을 사실상 제공 — ¥63만 지출 전에 2모델 분해 가설을 검증하는 최저비용 실험. 이것을 먼저 할 것.
- **TM01 텔레옵은 isomorphic(7-DoF→7-DoF)이라 관절공간 라벨이 무손실** — Model 2 라벨 무료, Model 1 라벨은 FK로 유도. VR 컨트롤러 rig(EE 포즈→IK로 관절 합성)보다 훨씬 유리.
- **수집·배포 측은 Linux 필수.** Mac은 학습·MuJoCo만; 로봇 통신·텔레옵 수집·배포 호스팅 불가. 배포 타깃은 탑재 Orin이므로 모델 사이징은 Orin 기준(M4/32GB는 학습·시뮬 제약). 참조점: GraspVLA는 L40s에서 ~9GB/200ms.
- **구매 전 세대·EE 구성 서면 확인.** 생산 G1은 흔히 비대칭(석션 1+그리퍼 1) — 한쪽이 석션이면 action 공간 절반이 이진 진공 토글이 되어 진짜 양팔 협조 태스크가 불가능해짐. 양팔 조작이 목적이면 듀얼 그리퍼(또는 덱핸드)를 명기 주문.
- **Galbot은 이 아키텍처를 발표한 적 없음 — 위험이자 기회.** 훔칠 것 2가지: (1) 단일 네트워크+보조 포즈 head가 강력한 baseline — 이것이 2모델 분리와 비기면 분리는 복잡성 값을 못 함; (2) 그들의 action 공간은 EE delta+chunking, 반면 SDK 예제·공개 데이터는 절대 관절 위치 — 의도적으로 선택하고 벤더 관례와의 괴리를 인지할 것.
- **라이선스**: 시뮬 자산(Apache-2.0)·컨버터(MIT)는 상업 안전; GraspVLA 코드·웨이트는 CC BY-NC 4.0 — 제품 탑재·파생 초기화 불가 *(참고: REV.2에서 상업 파인튜닝 불가로 확정)*; GalbotSDK 라이선스는 확인 필요. SynGrasp-1B 미공개라 사전학습 재현도 불가.
- 상업 경로는 실재하고 소규모 팀 접근 가능(FREE 티어+공개 자산으로 시작, PRO는 하드웨어 도착 후). 영업에 직접 물을 것: 리드타임, 비중국 구매, TM01 가격, head-camera extrinsic + F/T 장착 frame.

---

## 토픽 4 — Galbot 자체 발표 스택 (GraspVLA / GroceryVLA / AstraBrain / GalbotSDK)과 서드파티 2모델 스택에의 함의

### VERIFIED

| 사실 | 수치 | 출처 |
|---|---|---|
| GraspVLA 아키텍처: AR VLM + flow-matching action expert, CoT 'Progressive Action Generation'(PAG): bbox → grasp pose → action chunk | InternLM2 1.8B(학습됨); frozen DINOv2+SigLIP+학습 projector; bbox 8토큰, grasp pose 6토큰; 손실 = AR 토큰 손실 + flow-matching 손실 합 | 논문 arXiv:2505.03233 Sec.4, Fig.3 |
| 계층형 아님 — '레벨'은 한 forward pass 안의 토큰 단계 | 별도 상·하위 네트워크/제어 루프/포즈 인터페이스 없음; action expert는 VLM KV cache 직독; 상·하위를 다른 rate로 돌릴 방법 없음 | 논문 |
| action 공간은 chunked EE **delta**(Cartesian); 중간 grasp pose는 **robot base frame**(head/카메라 frame 아님) | proprioception 최근 2 timestep 토큰화; baseline 학습 chunk=4; 데이터 action 10Hz 저장 | 논문 |
| F/T·tactile 전혀 미사용 — 자체 실패 분석이 이를 실패 원인으로 지목 | 실패 분포: 매끄러운 표면 slip 21%("tactile feedback might help resolve") / 언어 모호 시 hesitation 31% / clutter 오인식 27% / 가림 14% / 조기 그리퍼 폐합·충돌 7% | 논문 App.M |
| 유효 정책 rate ~5Hz, latency 200ms, ~9GB VRAM (L40s) | vision 9ms / bbox 72ms / grasp pose 50ms / flow matching 64ms — AR CoT 토큰이 122/195ms로 지배적('reasoning'이 느림의 원인) | 논문 Table 3·15 + repo README |
| SynGrasp-1B: 10일간 RTX 4090 160장으로 생성 | 1e9 frames; 10,680 meshes / 240 categories(Objaverse LVIS); BoDex 파지 합성 + CuRobo 궤적 + MuJoCo 물리 검증 + Isaac Sim 레이트레이싱 렌더; ~38,400 GPU-hours; action 10Hz | 논문 Sec.3 |
| 사전학습에 실기체 action 데이터 0 (real:sim = 0:1) | 비합성 데이터는 GRIT(인터넷 grounding)뿐, bbox 토큰만 감독 | 논문 Sec.4 |
| 실기체는 단일팔 Franka Panda + 평행 그리퍼 — 명시적으로 G1 아님 | 외부 카메라 2대(D435 전방+D415i 측면); 작업공간 40×50×20cm; 핑거 +2cm 연장; 타 embodiment는 'future work' | 논문 Sec.7, 5.1 |
| zero-shot 실기체 ~93%, pi-0 대비 대폭 우위 | 합성 카테고리 basic 93.3/light 96.6/bg 93.3/dis 93.3/height 90.0, SPL 87.2; 웹 카테고리 93.3/90.0/93.3/86.6/86.6, SPL 84.7; 300 trials; pi-0 baseline 80.0(합성)/40.0(웹); Octo·OpenVLA 0-20% | 논문 Table 1 |
| few-shot post-training은 신규 행동당 10-100 데모 | 산업 부품 100 / 머그 핸들 제약 100 / 순차 파지 병당 10; 실 텔레옵 데이터는 post-training 단계에서만 사용 | 논문 Sec.5.4 |
| 공개 웨이트는 CC BY-NC 4.0(비상업), 파인튜닝 코드 미제공 | HF shengliangd/GraspVLA(+StereoVLA 변형); 추론 서버·시뮬 환경·실기 인터페이스만; SynGrasp-1B는 미공개(마지막 push 2025-12-29 기준) | 공식 repo |
| 실배포 하부는 고전 Cartesian **impedance 컨트롤러** — 손으로 짠 소뇌, 학습된 소뇌 아님 | Franka ROS+SERL 기반; Jacobian 변환+특이점 처리; receding horizon; 삼중 종속 1차 Butterworth 필터; 위치 기반 보간으로 latency jitter 흡수; 보고 실험은 전부 blocking 제어 | 논문 App.L |
| G1 양 손목 6축 F/T가 SDK에 노출 (재확인) | 2센서 × 6축; N/N·m/ns | 공식 SDK |
| 공식 SDK 센서 스위트: 스테레오 RGB 헤드(depth 없음), 손목 RGBD 2, LiDAR, 초음파, 토르소 IMU | 카메라 총 7; 손목 D405 1280×720(실 depth); 헤드 depth 없음 → foundation_stereo 예제가 유일한 인지 예제; get_synced_observation 시각 정렬 | 공식 SDK |
| 제어 관절 구조 23-D (재확인) | 5 leg + 2 head + 7+1 + 7+1; head는 pan/tilt 2-DoF — **head frame이 base 대비 움직임** | 공식 SDK + 컨버터 |
| SDK WBC가 양손 목표 포즈→협조 관절 운동을 이미 제공 — 'Model 2'가 해석적 형태로 실재 | 7-vector 포즈; lee/ree/head_pose; SingoriX 타깃은 관절+태스크 공간 궤적 혼합 가능; GalbotMotion: IK/FK/Jacobian/플래닝/충돌검사/장애물·툴 부착 | 공식 SDK |
| 125Hz 모델 출력 스트리밍 명시 지원 + 금지 API 경고 | 궤적 기본 dt=0.008s(125Hz); set_joint_positions는 모델 추론 출력에 "NOT suitable" 명시 경고; JointCommand에 position/velocity/acceleration/effort/Kp/Kd — 토크·임피던스류 플러밍은 도달 가능 위치에 존재 | 공식 SDK |
| LeRobot을 서드파티 학습 포맷으로 공인 — 단 컨버터가 F/T를 폐기 | MIT; 23-D state + 23-D action(다음-상태 타깃 = 관절공간 위치, delta도 Cartesian도 아님); AV1; F/T·tactile 미포함 | 공식 (galbot-mcap2lerobot) |
| GalbotSDK Apache-2.0·상업 사용 가능, Linux 바이너리 전용 *(주: 토픽 2·3 조사는 NOASSERTION으로 판정 — 토픽 간 상충, 라이선스 파일 직접 확인 필요)* | v1.9.1(2026-07-03); Ubuntu 20-24; 바이너리 2타깃(linux-x86_64/aarch64) | 공식 repo |
| 현행 플래그십 'AstraBrain'(银河星脑): 통합 end-to-end 대뇌-소뇌-신경제어 아키텍처 주장, 모듈 분리를 명시적으로 거부 | "打破了业界将规划、控制、操作模块割裂的传统" (계획·제어·조작을 모듈로 분리하는 업계 전통을 깼다); WAIC 2026 아침식사(토스트·음료) 양팔 장시계 시연+외란 복구; **논문·모델 크기·rate·인터페이스 스펙 전무** | 서드파티 (sina, sohu, chinadaily, 163 — 회사 발표 보도) |
| GroceryVLA는 GraspVLA 위에 구축된 폐쇄형 상용 모델 | 리테일용 첫 end-to-end VLA 주장; 수백 SKU(연포장·유리·약품·컵); 논문·웨이트·arXiv 없음 | 서드파티 + 홍보자료 |
| 배포 플릿 수치(약국/리테일) | 매장 주문 ~100건·베이징 10+ 가동; 매장 ~40m²·5,000+ SKU·~6,000 슬롯·로봇 1대·24/7; >1년 연속 가동 주장; 피크 370주문/일; >95% 파지 성공; 커피 시연 >99.97%(5일 수천 잔); 2025 세계 휴머노이드 로봇 대회 약품 분류 금메달(170점, 22팀 대부분 텔레옵인 가운데 완전 자율); 매장당 노동 >50% 절감 주장; 소매가 ~63만 RMB | 서드파티 (21jingji, bjnews, huxiu, PRNewswire) |
| Galbot 오픈소스 포트폴리오는 locomotion/내비/추적 — 조작 아님, 대부분 자사 G1용도 아님 | 공개 repo 13; 최고 758★(OpenTrack); OpenWBT(754★)는 **Unitree** G1/H1용; Galbot G1용 공개 양팔 조작 정책 0 | 공식 GitHub org |
| G1 헤드라인 스펙의 서드파티 수치는 상호 모순 | 47 DoF / 핸드 12-DoF(또는 22-DoF SharpaWave) / 1.68-1.73m / 55-85kg / 페이로드 5·10·20kg / 배터리 4-10h / 5,000+ 물체 95-97% — 상충 다수 | 서드파티 |

주요 출처: arXiv:2505.03233 · https://github.com/PKU-EPIC/GraspVLA · pku-epic.github.io/GraspVLA-web · https://github.com/GalaxyGeneralRobotics/GalbotSDK · https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot

### UNVERIFIED (찾아봤으나 확인 실패 — 벤더 문의 목록)

- **GraspVLA 자체의 action chunk 길이.** baseline은 chunk 4로 학습됐다고 명시되나 GraspVLA 본인의 chunk 크기는 미기재(5Hz 정책 vs 10Hz 저장 action에서 추론당 2+ 소비로 추정될 뿐).
- **Franka 벤치 외 배포 제어 주파수.** G1에서의 어떤 VLA 제어 rate도 미발표.
- **GraspVLA가 G1에서 실행된 적이 있는지.** 논문은 Franka 한정·타 embodiment는 future work. 마케팅은 'G1에 GraspVLA 탑재'를 암시하나 공개 모델=배포 모델임을 확증하는 기술 소스 없음.
- **AstraBrain의 기술 세부 일체.** 논문·arXiv·모델 크기·파라미터 수·제어 rate·학습 데이터 규모·sim/real 비율·latency·벤치마크 전무 — 전부 중국 언론의 회사 발표 보도.
- **AstraBrain의 大脑/小脑/신경제어 레벨 간 인터페이스.** 이 프로젝트에 가장 직결된 질문인데 완전 미발표. 명칭은 3레벨을 시사하나 경계를 넘는 것(포즈? latent? 토큰?)·rate·개별 학습 여부 불명. 'unified end-to-end'라는 표현은 경계가 아키텍처가 아닌 개념적일 가능성 시사.
- **Galbot의 System-1/System-2 framing 발표 여부.** 快系统/慢系统·系统1/系统2로 영·중 검색 — first-party 기술 진술 없음, 일반 언론 프레이밍뿐.
- **GroceryVLA의 아키텍처·모델 크기·action 공간·제어 rate·데이터 규모·real:sim 비율.** 논문 없음. 유일한 아키텍처 주장은 'GraspVLA 위에 구축'.
- **'GraspVID'.** 존재 확인 불가 — GraspVLA 혼동이거나 타 그룹 비디오 사전학습 작업과의 혼동 가능성.
- **'GroceryVLM'.** 실명은 GroceryVLA(Vision-Language-**Action**). GroceryVLM은 발견 못 함.
- **'一脑多形'(one-brain-many-bodies) 슬로건의 기술 주장 결부.** Galbot 메시징은 태스크·물체 일반화 중심; cross-embodiment는 인접 PKU-EPIC 작업(LDA-1B)에 주로 등장. 수치가 딸린 Galbot cross-embodiment 제품 주장 미확인.
- **배포 매장의 사람 개입률.** 영·중 어느 소스에도 보고 없음. '>95%'·'>99.97%'는 파지 시도 성공률이지 태스크 완료율이 아니며, 사람이 리셋·재고 보충·예외 처리를 얼마나 하는지 무언급. 24/7 매장에서 정작 중요한 수치가 부재.
- **약국 배포의 픽당/주문당 사이클 타임.** '피크 370주문/일'이 유일한 처리량 프록시.
- **>95% 파지와 약국 배포가 GraspVLA/GroceryVLA/AstraBrain/고전 파이프라인 중 무엇으로 도는지.** 고정 6,000슬롯 선반이면 상당 부분 스크립트 모션+인식으로 가능; 공개 주장은 구분하지 않고, SDK의 고전 모션플래닝 스택으로도 충분할 수 있음.
- **GraspVLA/GroceryVLA/AstraBrain의 서드파티 라이선스 가능 여부·조건·가격.** 공개 웨이트는 CC BY-NC(상업 불가); 상용 모델은 외부 문서 0; 개발자용 모델 API·마켓플레이스·가격 없음.
- **Galbot이 G1 고객에게 파인튜닝 서비스/post-training 파이프라인을 제공하는지.** GraspVLA repo는 추론 전용.
- **Galbot이 자사 학습 모델 어디에서든 손목 F/T를 쓰는지.** 하드웨어·SDK는 있으나 GraspVLA는 미사용, 자사 컨버터도 폐기 — force를 소비하는 Galbot 모델을 발견 못 함.
- **G1 탑재 컴퓨트(SoC·GPU·VRAM).** 이 토픽의 조사 소스에서는 미확인(aarch64 바이너리로 ARM 추정만). *(참고: 토픽 1·2의 공식 스펙에서 AGX Orin 64GB 275 TOPS로 확정, REV.2에서 추론은 탑재 Orin으로 결정)*
- **헤드 카메라 해상도·intrinsics.** 손목은 D405 1280×720 확인; 헤드 네이티브 스테레오 해상도·baseline 미문서.
- **FoundationStereo 헤드-depth 예제의 실시간 성능.** latency·rate 미기재 — 헤드에 depth 센서가 없어 중요.
- **SynGrasp-1B 공개 여부.** 'Coming Soon' 지속 — 사전학습 코퍼스 재사용·확장 불가.
- **상충 스펙의 정합.** 55 vs 85kg, 5/10/20kg, 12 vs 22-DoF, 4 vs 10h — galbot.com 본진은 JS 렌더링이라 1차 스펙시트 추출 실패.
- **Galbot G1용 first-party 텔레옵 수집 rig 존재 여부.** OpenWBT는 Unitree용; galbot-mcap2lerobot의 존재가 내부 기록의 존재를 증명하나 기록 도구 자체는 비공개. *(참고: 토픽 3에서 TM01 leader-follower rig가 공식 제품으로 확인됨)*

### IMPLICATIONS (핵심 압축)

- **최대 발견: Model 2는 해석적 형태로 이미 출하되어 있음.** SDK WBC가 제안된 Model 1 출력(양손 7-vector 포즈)을 그대로 받아 IK/FK/Jacobian/충돌검사 기반의 협조 전신 관절 운동으로 변환하며 125Hz 스트리밍 가능. 벤더 연구진 스스로도 GraspVLA에서 소뇌를 학습하지 않고 임피던스 컨트롤러+Butterworth 필터를 썼음. 즉 "벤더가 출하하고 벤더 연구진도 학습하지 않기로 한 컴포넌트의 신경망 대체"가 계획의 최약점 — 우선 삭제하고, Model 1을 기존 WBC에 대고 세운 뒤 해석적 해가 명백히 실패하는 지점에서만 학습 Model 2로.
- **쐐기는 force이고, 실재함.** G1에 양 손목 6축 F/T가 있는데 Galbot의 어떤 발표 모델도 force를 소비하지 않고, 공인 데이터 경로는 그것을 조용히 폐기하며, GraspVLA 실패의 21%가 tactile로 해결 가능한 slip. 접촉 중심·force-aware·양팔 조작이 Galbot 발표물 전체의 진짜 공백 — '더 나은 VLA'가 아니라 이것이 방어 가능한 wedge. 기록 스키마 확장은 직접 해야 함(23-D 벡터에 슬롯 없음).
- **frame 선택은 자초한 문제.** GraspVLA는 base frame, SDK는 world frame; head는 2-DoF pan/tilt로 시변이고 depth 센서도 없음(FoundationStereo가 제3의 네트워크로 필요). 손목엔 진짜 D405 depth가 있음. 권고: 벤더처럼 base/world frame으로 예측하고 head-frame 세금 납부 중단.
- **M4 Mac mini는 이중 hard blocker.** (1) SDK는 Linux 바이너리 전용 — Mac은 Linux 브리지 없이 로봇과 물리적으로 통신 불가; (2) GraspVLA는 L40s에서 1.8B·200ms·9GB로 5Hz — 그것도 단일팔 파지 하나. 현실적 예산: Model 1은 1-5Hz, Model 2(또는 벤더 WBC)는 100+Hz — 정확히 SDK가 설계된 2-rate 분리. CoT 토큰이 122/195ms라는 점도 주의: Model 1이 VLM이면 그 비용을 상속. *(참고: REV.2에서 추론은 탑재 AGX Orin, REV.3에서 학습·시뮬은 RTX 3090/Ubuntu 박스로 확정)*
- **from scratch 학습 금지 — 벤더 자신의 수치가 반대 논거.** SynGrasp-1B는 ~38,400 GPU-hours로 만든 단일팔·그리퍼·탁상 파지(무force). 반면 few-shot post-training은 100/100/10 데모로 동작 — 이것이 실행 가능한 교훈. 시뮬-퍼스트는 정신은 옳으나 파운데이션 모델이 아니라 좁은 양팔 접촉 태스크를 겨냥하고, 실 텔레옵 예산은 수백 데모 규모의 적응용으로.
- **벤더 모델은 상업적으로 사용 불가.** GraspVLA는 CC BY-NC+추론 전용+Franka+데이터 미공개; GroceryVLA·AstraBrain은 논문도 API도 없음. 상업적으로 깨끗한 것: GalbotSDK와 컨버터(MIT). 공식 MCAP→LeRobot 컨버터의 존재가 시사하는 의도된 서드파티 경로: G1에서 텔레옵 수집 → LeRobot 변환 → **열린 모델**(pi-0 / SmolVLA / GR00T급) 파인튜닝. 벤더 데이터 포맷으로, 벤더 WBC에 대고, 열린 모델을 파인튜닝하는 것이 실용적 답.
- **벤더는 아키텍처적으로 반대 베팅 중 — 단 발표된 근거는 0.** AstraBrain은 2모델 분리를 초월했다고 마케팅하지만 논문·수치·인터페이스 스펙이 없고, Galbot이 실제 발표한 GraspVLA는 내부적으로 단계화(bbox→pose→action)되고 외부적으로 수제 임피던스 컨트롤러와 짝지어져 있음 — 마케팅보다 제안된 분해에 가까움. 마케팅 때문에 분리를 버리지 말고, 하부가 이미 출하됐다는 이유로 분리를 재고할 것.
- **PAG CoT가 훔칠 설계 아이디어이고 Model 1의 형태를 검증해 줌.** 인터넷 grounding 데이터로도 감독 가능한 중간 공간 타깃(bbox)을 먼저 내게 한 것이 240 카테고리 학습 정책을 웹 카테고리 93.3%(pi-0는 40.0)로 끌어올림. Model 1("장면을 보고 손이 갈 곳을 결정")은 구조적으로 같은 수 — 2레벨 설계의 가장 강한 찬성 논거이며, Model 1을 인터넷 grounding/포즈 데이터와 co-training하라는 구체적 전술을 가리킴. 단 비용 주의(CoT가 GraspVLA 195ms 중 122ms).
- **배포 수치를 액면 그대로 믿지 말 것.** 전부 파지 시도 성공률/처리량이지 태스크 완료율이 아니고, 개입률은 어느 언어로도 미보고. 고정 planogram 약국은 SDK 고전 플래너로도 상당히 풀리는 구조화 환경. 실제 상업 기준은 비구조 장면의 예외율·개입률 — 아무도 발표하지 않은 영토이고 force-aware 양팔 시스템이 진짜 이길 수 있는 곳.
- **실용 시퀀스**: (1) 학습 Model 2 일단 삭제, WBC 사용하되 Model 1 출력 계약(양손 7-vector)은 유지해 나중에 같은 인터페이스로 교체 가능하게; (2) 로봇 쪽에 Linux 박스, M4는 저rate Model 1 호스트로; (3) galbot-mcap2lerobot을 즉시 fork해 **수집 전에** 손목 F/T 채널 추가; (4) head-image frame 대신 base/world frame; (5) sim-to-real 야망은 GraspVLA식 randomization의 좁은 양팔 접촉 태스크로 사이징, 실 데모는 100 규모; (6) 라이선스상 Galbot 모델이 아닌 열린 LeRobot 호환 정책을 파인튜닝한다고 가정.
