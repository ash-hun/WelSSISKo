from transformers import pipeline

REPO = 'Ash-Hun/WelSSiSKo'

q1 = "BLOOM은 몇 개의 프로그래밍 언어를 지원합니까?"
c1 = "BLOOM은 1,760억 개의 매개변수를 보유하고 있으며 46개 언어 자연어와 13개 프로그래밍 언어로 텍스트를 생성할 수 있습니다."

q2 = "안녕?  내 이름은 Ash야. 너의 취미는 뭐니?"
c2 = "안녕 나를 소개하지 이름 김하온 직업은 travler 취미는 tai chi, meditation, 독서, 영화시청 랩 해 터 털어 너 그리고 날 위해 증오는 빼는 편이야 가사에서 질리는 맛이기에 나는 텅 비어 있고 prolly 셋 정도의 guest 진리를 묻는다면 시간이 필요해 let me guess 아니면 너의 것을 말해줘 내가 배울 수 있게 난 추악함에서 오히려 더 배우는 편이야 man"


question_answerer = pipeline("question-answering", model=REPO)
print(question_answerer(question=q1, context=c1, truncation=True))
print(question_answerer(question=q2, context=c2, truncation=True))