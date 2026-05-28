import os
import base64
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from prompts import SYSTEM_PROMPT

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

client = OpenAI(api_key=api_key)

MODEL = "gpt-5.4"
# model_not_found 뜨면 임시로 아래 사용
# MODEL = "gpt-4.1-mini"

st.set_page_config(page_title="Socratic Tutor", layout="wide")
st.title("케이스 기반 Socratic 사고 촉진 GPT")


# =========================
# 세션 상태
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "class_started" not in st.session_state:
    st.session_state.class_started = False

if "phase" not in st.session_state:
    st.session_state.phase = "opening"

if "expansion_question_count" not in st.session_state:
    st.session_state.expansion_question_count = 0

if "case_presented" not in st.session_state:
    st.session_state.case_presented = False


# =========================
# 교사용 입력
# =========================
st.sidebar.header("교사용 사전 입력")

subject = st.sidebar.selectbox(
    "교과 영역",
    ["경제", "윤리", "법", "정치", "지리", "일반사회"]
)

activity_type = st.sidebar.selectbox(
    "활동 유형",
    ["토론형", "개념 이해형", "탐구형"]
)

topic = st.sidebar.text_input("논의 주제")
goal = st.sidebar.text_area("학습 목표")

uploaded_image = st.sidebar.file_uploader(
    "(선택) 교재/통계자료 이미지 업로드",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    st.sidebar.image(uploaded_image, caption="업로드된 자료")

teacher_context = f"""
[교사용 사전 입력]
교과 영역: {subject}
활동 유형: {activity_type}
논의 주제: {topic}
학습 목표: {goal}
"""


# =========================
# 기본 함수
# =========================
def image_to_data_url(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def build_conversation_text(max_messages=14):
    text = ""
    for msg in st.session_state.messages[-max_messages:]:
        role = "학생" if msg["role"] == "user" else "GPT"
        text += f"{role}: {msg['content']}\n"
    return text


def add_assistant_message(text):
    st.session_state.messages.append(
        {"role": "assistant", "content": text}
    )


# =========================
# 근거 판단
# =========================
def judge_reason(student_input, phase):
    """
    학생 답변에 근거가 충분한지 GPT에게 판단시킴.
    """
    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": """
너는 학생 답변에 근거가 충분히 드러났는지 판별하는 보조 판별기다.
반드시 아래 둘 중 하나만 출력한다.

ENOUGH
INSUFFICIENT

판단 기준:
- 단순 동의, 단순 입장, "응", "그런 것 같아", "찬성", "반대"만 있으면 INSUFFICIENT
- 주장이나 생각만 있고 이유가 없으면 INSUFFICIENT
- 왜 그렇게 생각하는지 이유, 사례, 조건, 가치 기준, 현실적 근거가 있으면 ENOUGH
- 짧아도 명확한 이유가 있으면 ENOUGH
- 길어도 결론 반복뿐이면 INSUFFICIENT
"""
                },
                {
                    "role": "user",
                    "content": f"""
현재 흐름: {phase}
논의 주제: {topic}

학생 답변:
{student_input}
"""
                }
            ]
        )

        result = response.output_text.strip().upper()
        return "ENOUGH" in result

    except Exception:
        # 판별기 오류로 앱이 멈추지 않게 기본은 ENOUGH 처리
        return True


# =========================
# GPT 호출
# =========================
def call_gpt(user_text, phase, use_web=False, require_question=True):
    phase_instruction = {
        "opening": """
첫 발화다.

반드시 지킬 것:
- 실제 사례, 뉴스 링크, 판례, 정책 사례를 제시하지 않는다.
- 웹검색한 듯한 내용을 말하지 않는다.
- 주제와 관련된 일반적인 현실 맥락만 2~3문장으로 자연스럽게 제시한다.
- 그다음 학생의 초기 입장을 묻는 질문 1개로 끝낸다.
- "현실 맥락을 말할게", "수업을 시작할게", "다음 단계로 넘어갈게" 같은 메타 발화를 하지 않는다.

토론형이면:
- 찬성 / 반대 / 조건부 입장 중 하나로 말해도 된다고 짧게 덧붙일 수 있다.

개념 이해형이면:
- 현상을 보고 왜 그런 일이 나타나는지 묻는다.

탐구형이면:
- 어떤 원인들이 작용했을지 묻는다.
""",

        "reason_check": """
학생 답변에 근거가 부족하다.
학생의 입장이나 생각을 1문장으로 짧게 인정한다.
왜 그렇게 생각하는지 근거를 1개만 물어본다.
반대 관점, 현실 사례, 개념 정리, 마무리로 넘어가지 않는다.
반드시 질문 1개로 끝낸다.
""",

        "structure": """
학생이 입장과 근거를 제시했다.
학생 답변을 짧게 인정하고 요약한다.
핵심 근거, 판단 기준, 가치나 전제를 자연스럽게 구조화한다.
시연 대화처럼 "네 생각은 ~라는 쪽이야", "핵심은 ~야" 정도로 정리한다.
마지막에는 반대 관점, 예외 상황, 이해관계자 차이, 판단 기준의 모호함 중 하나를 골라 질문 1개로 끝낸다.
현실 사례는 아직 제시하지 않는다.
""",

        "expand": """
관점 확장 단계다.
학생이 답한 내용을 짧게 인정하고 정리한다.
기존 입장이 어떻게 더 정교해졌는지 보여준다.
아직 실제 사례를 자세히 제시하지 않는다.

관점 확장 질문은 전체 최대 3개까지만 가능하다.
현재까지 관점 확장 질문 수를 참고해서, 3개 미만이면 추가 질문 1개로 끝낼 수 있다.
이미 충분히 확장되었거나 3개에 도달했다면 질문을 반복하지 말고 실제 사례를 보면 기준이 더 선명해질 수 있다는 흐름으로 연결한다.
""",

        "case": """
이제 실제 사례를 제시하는 단계다.
학생의 기존 답변과 직접 연결되는 실제 사례를 웹에서 찾아 자연스럽게 제시한다.

반드시 포함:
- 구체적인 사건명, 국가/지역, 시기, 정책명 또는 판결명
- 출처명과 링크
- 가능하면 통계나 자료 출처
- 학생의 기존 기준이 이 사례에서 어떻게 유지되거나 흔들리는지
- 이해관계가 충돌하는 지점
- 마지막에는 학생이 자기 기준을 사례에 적용해보는 질문 1개

금지:
- "미국 일부 주", "최근 기사", "어떤 나라"처럼 모호하게 말하지 않는다.
- "사례 제목:", "쟁점:" 같은 카드식/표식 표현을 쓰지 않는다.
- 사례 설명 후 바로 교과 개념 강의로 넘어가지 않는다.
- 출처 없는 사례를 만들지 않는다.

말투:
- 시연 대화처럼 자연스럽게 설명한다.
- 딱딱한 보고서 문체를 피한다.
""",

        "case_reflection": """
학생이 실제 사례에 대해 해석했다.
학생 답변을 짧게 인정하고, 사례 속 갈등을 어떻게 이해했는지 구조화한다.
학생의 기존 기준이 사례를 통해 어떻게 유지되거나 수정되었는지 보여준다.
바로 마무리하지 말고 교과 개념과 연결될 수 있도록 자연스럽게 이어간다.
마지막에는 필요하면 학생의 생각 변화를 묻는 질문 1개로 끝낸다.
""",

        "concept": """
학생 답변을 교과 개념 1~2개와 연결한다.
개념을 길게 설명하지 말고, 학생이 말한 내용이 어떤 사회과학 개념과 이어지는지 짧게 보여준다.
예: 기본권 충돌, 자기결정권, 생명권, 절차적 정당성, 형평성과 효율성, 상호의존, 국제 분업 등.
마지막에는 학생이 자신의 생각 변화를 돌아보게 하는 질문 1개로 끝낸다.
""",

        "summary": """
오늘 대화 흐름을 짧게 메타인지적으로 정리한다.
포함할 것:
- 처음 생각
- 중간에 새로 발견한 기준
- 생각이 확장된 지점
- 아직 남은 질문

그다음 협동 토론 질문 1~2개를 제안한다.
부담스럽게 길게 쓰지 않는다.
"""
    }.get(phase, "")

    question_rule = """
[질문 규칙]
- 반드시 마지막은 학생이 답할 수 있는 질문 1개로 끝낸다.
- 질문은 한 번에 여러 개 나열하지 않는다.
- 질문 문장은 물음표(?)로 끝낸다.
""" if require_question and phase != "summary" else """
[질문 규칙]
- 이번 응답에서는 질문을 반드시 할 필요는 없다.
- 질문을 한다면 1개만 한다.
"""

    expand_rule = ""
    if phase == "expand":
        expand_rule = f"""
[관점 확장 질문 수]
현재까지 관점 확장 질문 수: {st.session_state.expansion_question_count}
관점 확장 질문은 최대 3개까지만 한다.
"""

    content = [
        {
            "type": "input_text",
            "text": f"""
아래 수업을 진행한다.

{teacher_context}

[현재 내부 흐름]
{phase}

[이번 응답 지시]
{phase_instruction}

{expand_rule}

{question_rule}

[공통 응답 규칙]
- 반말을 사용한다.
- 학생에게 내부 흐름명이나 단계명을 말하지 않는다.
- "이제 다음 단계로", "현실 맥락을 말할게", "네 답변을 듣고 나면" 같은 메타 발화를 피한다.
- 질문만 던지지 말고, 학생 답변을 짧게 요약하거나 구조화한다.
- 답변은 보통 3~8문장 정도로 한다.
- 학생이 말하지 않은 근거를 너무 많이 대신 만들어내지 않는다.
- 시연 대화처럼 자연스럽게: 인정/요약 → 사고 구조화 → 필요한 질문 흐름을 따른다.
- 현실 사례는 phase가 case일 때만 제시한다.
- phase가 opening, structure, expand일 때는 실제 뉴스, 판례, 정책 사례, 링크를 제시하지 않는다.
- 현실 사례를 제시할 때는 구체적 사건명, 국가, 시기, 출처명과 링크를 포함한다.
- 링크는 절대 중간에 끊지 않는다.
- 교재나 통계 이미지가 업로드되어 있다면 필요할 때 참고한다.

[현재까지의 대화]
{build_conversation_text()}

[마지막 입력]
{user_text}
"""
        }
    ]

    if uploaded_image:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(uploaded_image)
            }
        )

    tools = []
    if use_web:
        tools = [{"type": "web_search"}]

    try:
        response = client.responses.create(
            model=MODEL,
            tools=tools,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        )

        return response.output_text.strip()

    except Exception as e:
        return f"API 오류가 발생했어: {e}"


def ask_for_reason(student_input, phase):
    return call_gpt(
        user_text=f"""
학생 답변에 아직 근거가 충분히 드러나지 않았다.
다음 흐름으로 넘어가지 마.
학생의 답변을 짧게 인정하고, 왜 그렇게 생각하는지 근거를 1개만 물어봐.
반대 관점, 현실 사례, 개념 정리, 마무리로 넘어가지 마.

학생 답변:
{student_input}
""",
        phase="reason_check",
        use_web=False,
        require_question=True
    )


def present_case():
    case_reply = call_gpt(
        user_text="""
학생의 기존 답변과 직접 연결되는 실제 사례를 웹에서 찾아 자연스럽게 제시해.
구체적인 사건명, 국가, 시기, 정책명 또는 판결명을 포함해.
출처명과 링크를 포함해.
가능하면 관련 통계나 자료 출처도 함께 제시해.
표나 카드 형식은 쓰지 말고, 시연 대화처럼 부드럽게 설명해.
마지막에는 학생이 자기 기준을 사례에 적용해보는 질문 1개로 끝내.
""",
        phase="case",
        use_web=True,
        require_question=True
    )

    add_assistant_message(case_reply)
    st.session_state.phase = "case_reflection"
    st.session_state.case_presented = True


# =========================
# 사이드바 버튼
# =========================
st.sidebar.divider()
st.sidebar.subheader("수업 진행")

if st.sidebar.button("수업 시작"):
    if not topic or not goal:
        st.sidebar.warning("논의 주제와 학습 목표를 입력해줘.")
    else:
        st.session_state.messages = []
        st.session_state.class_started = True
        st.session_state.phase = "opening"
        st.session_state.case_presented = False
        st.session_state.expansion_question_count = 0

        opening = call_gpt(
            user_text="학생에게 바로 보여줄 첫 발화를 작성해. 실제 사례나 링크는 절대 제시하지 마.",
            phase="opening",
            use_web=False,
            require_question=True
        )

        add_assistant_message(opening)
        st.session_state.phase = "structure"
        st.rerun()


if st.sidebar.button("현실 사례 연결"):
    if not st.session_state.class_started:
        st.sidebar.warning("먼저 수업을 시작해줘.")
    else:
        present_case()
        st.rerun()


if st.sidebar.button("마무리 정리"):
    if not st.session_state.class_started:
        st.sidebar.warning("먼저 수업을 시작해줘.")
    else:
        summary = call_gpt(
            user_text="오늘 대화 흐름과 학생의 사고 변화를 짧게 정리하고, 협동 토론 질문 1~2개를 제안해.",
            phase="summary",
            use_web=False,
            require_question=False
        )
        add_assistant_message(summary)
        st.session_state.phase = "done"
        st.rerun()


if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.class_started = False
    st.session_state.phase = "opening"
    st.session_state.case_presented = False
    st.session_state.expansion_question_count = 0
    st.rerun()


# =========================
# 본문
# =========================
st.subheader("학생 대화 화면")

with st.expander("현재 수업 설정 보기"):
    st.write(f"교과 영역: {subject}")
    st.write(f"활동 유형: {activity_type}")
    st.write(f"논의 주제: {topic}")
    st.write(f"학습 목표: {goal}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================
# 학생 입력 처리
# =========================
student_input = st.chat_input("학생 답변을 입력해")

if student_input:
    if not st.session_state.class_started:
        st.warning("먼저 왼쪽에서 수업 정보를 입력하고 '수업 시작'을 눌러줘.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": student_input}
        )

        phase = st.session_state.phase

        # 근거 필수 phase
        reason_required_phases = [
            "structure",
            "expand",
            "case_reflection",
            "concept"
        ]

        if phase in reason_required_phases:
            if not judge_reason(student_input, phase):
                reply = ask_for_reason(student_input, phase)
                add_assistant_message(reply)
                st.session_state.phase = phase
                st.rerun()

        # 1. 입장/근거 구조화 단계
        if phase == "structure":
            reply = call_gpt(
                user_text=f"""
학생이 입장과 근거를 제시했다.
학생 답변을 짧게 요약하고, 핵심 근거와 판단 기준을 구조화해.
마지막에는 반대 관점, 예외 상황, 이해관계자 차이, 판단 기준의 모호함 중 하나를 골라 질문 1개로 끝내.
현실 사례나 링크는 아직 제시하지 마.

학생 답변:
{student_input}
""",
                phase="structure",
                use_web=False,
                require_question=True
            )

            add_assistant_message(reply)
            st.session_state.phase = "expand"
            st.session_state.expansion_question_count = 1
            st.rerun()

        # 2. 관점 확장 단계
        elif phase == "expand":
            # 학생이 방금 관점 확장 질문에 답한 상태
            if st.session_state.expansion_question_count < 3:
                reply = call_gpt(
                    user_text=f"""
학생이 관점 확장 질문에 답했다.
학생 답변을 짧게 정리하고, 기존 생각이 어떻게 넓어졌는지 보여줘.
아직 현실 사례를 제시하지 마.
다른 관점, 예외 상황, 이해관계자 차이, 판단 기준의 모호함 중 하나만 골라 추가 질문 1개로 끝내.
같은 질문을 반복하지 마.

학생 답변:
{student_input}
""",
                    phase="expand",
                    use_web=False,
                    require_question=True
                )

                add_assistant_message(reply)
                st.session_state.expansion_question_count += 1
                st.session_state.phase = "expand"
                st.rerun()

            else:
                bridge = call_gpt(
                    user_text=f"""
학생이 관점 확장 단계에서 충분히 답했다.
학생 답변을 짧게 정리하고, 기존 입장이 어떻게 정교해졌는지 보여줘.
질문을 반복하지 마.
실제 사례를 보면 이 기준이 현실에서 어디서 흔들리는지 더 선명해질 수 있다는 흐름으로 자연스럽게 연결해.
아직 사례명, 링크, 판례, 정책명은 제시하지 마.

학생 답변:
{student_input}
""",
                    phase="expand",
                    use_web=False,
                    require_question=False
                )

                add_assistant_message(bridge)
                present_case()
                st.rerun()

        # 3. 실제 사례에 대한 학생 해석
        elif phase == "case_reflection":
            reply = call_gpt(
                user_text=f"""
학생이 현실 사례에 대해 해석했다.
학생 답변을 짧게 정리하고, 사례 속 갈등을 어떻게 이해했는지 구조화해.
학생의 기준이 사례를 통해 어떻게 유지되거나 수정되었는지 보여줘.
그다음 교과 개념과 연결될 수 있도록 자연스럽게 이어가.
마지막에는 학생 생각의 변화나 기준을 묻는 질문 1개로 끝내.

학생 답변:
{student_input}
""",
                phase="case_reflection",
                use_web=False,
                require_question=True
            )

            add_assistant_message(reply)

            concept_reply = call_gpt(
                user_text="""
방금까지의 대화와 학생 답변을 바탕으로 교과 개념 1~2개와 연결해 짧게 정리해.
마지막에는 학생이 자신의 생각 변화를 돌아보는 질문 1개로 끝내.
""",
                phase="concept",
                use_web=False,
                require_question=True
            )

            add_assistant_message(concept_reply)
            st.session_state.phase = "concept"
            st.rerun()

        # 4. 개념 연결 이후 학생 답변
        elif phase == "concept":
            summary = call_gpt(
                user_text=f"""
학생이 자신의 생각 변화를 답했다.
오늘 대화 흐름을 짧게 정리하고, 협동 토론 질문 1~2개를 제안해.

학생 답변:
{student_input}
""",
                phase="summary",
                use_web=False,
                require_question=False
            )

            add_assistant_message(summary)
            st.session_state.phase = "done"
            st.rerun()

        # 5. 완료 이후 자유 대화
        else:
            reply = call_gpt(
                user_text=student_input,
                phase="summary",
                use_web=False,
                require_question=False
            )
            add_assistant_message(reply)
            st.rerun()