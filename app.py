import os
import re
from collections import Counter

# Matplotlib 웹 서버 전용 비대화형 백엔드 설정 (무한 로딩 방지)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import seaborn as sns
import streamlit as st


# ==========================================
# 0. 한글 폰트 설정 함수
# ==========================================
def setup_korean_font():
    """폰트 등록 및 마이너스 기호 깨짐 방지"""
    font_path = "NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 텍스트 감정 분석 및 빈도 추출 함수
# ==========================================
def analyze_opinion_words(text_series):
    """한줄 의견 텍스트에서 단어를 추출하고 상위 10개 빈도를 계산합니다."""
    positive_keywords = {
        "좋다", "좋음", "만족", "최고", "유익", "추천", "즐거움", "보람", "알참",
        "편리", "쾌적", "친절", "재미", "성장", "도움", "완벽", "행복", "깨끗"
    }
    negative_keywords = {
        "힘듦", "어려움", "불만", "부족", "최악", "개선", "불편", "지침", "피곤",
        "아쉬움", "부담", "답답", "낙후", "과제", "스트레스", "복잡", "지각", "불친절"
    }

    full_text = " ".join(text_series.dropna().astype(str))
    words = re.findall(r'[가-힣]{2,}', full_text)

    if not words:
        return pd.DataFrame()

    word_counts = Counter(words)

    analyzed_data = []
    for word, count in word_counts.most_common(50):
        if word in positive_keywords:
            label = "긍정"
        elif word in negative_keywords:
            label = "부정"
        else:
            label = "기타"

        analyzed_data.append({"단어": word, "빈도수": count, "구분": label})
        if len(analyzed_data) >= 10:
            break

    return pd.DataFrame(analyzed_data)


# ==========================================
# 2. 데이터 전처리 함수 (숫자형 강제 변환 적용)
# ==========================================
def preprocess_data(df):
    """문자열 형태의 수치 데이터를 숫자로 강제 변환 후 결측치 및 비정상 범위 필터링"""
    initial_count = len(df)
    cleaned_df = df.copy()

    # 숫자 데이터여야 하는 열 목록
    numeric_cols = ['공부시간', '수면시간', '스마트폰시간', '학교생활만족도']

    # 문자가 섞여 있으면 숫자로 강제 변환 (변환 불가한 문자는 NaN 처리)
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    # 1. 결측치(NaN) 제거
    cleaned_df = cleaned_df.dropna(subset=[col for col in numeric_cols if col in cleaned_df.columns] + ['학과']).copy()

    # 2. 유효 범위 필터링 (시간: 0~24시간, 만족도: 1~5점)
    time_cols = ['공부시간', '수면시간', '스마트폰시간']
    for col in time_cols:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df[(cleaned_df[col] >= 0) & (cleaned_df[col] <= 24)]

    if '학교생활만족도' in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df['학교생활만족도'] >= 1) & (cleaned_df['학교생활만족도'] <= 5)]

    final_count = len(cleaned_df)
    dropped_count = initial_count - final_count

    return cleaned_df, initial_count, dropped_count


# ==========================================
# 3. 메인 앱 실행 함수
# ==========================================
def main():
    st.set_page_config(page_title="캠퍼스 생활 설문조사 분석", layout="wide")
    st.title("🎓 캠퍼스 생활 설문조사 분석 대시보드")

    setup_korean_font()

    # 1. 파일 업로드
    st.header("1. 설문조사 데이터 파일(CSV) 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding='cp949')

        # 컬럼 공백 제거 및 열 이름 표준화
        raw_df.columns = raw_df.columns.str.strip()
        column_rename_map = {
            '만족도': '학교생활만족도',
            '학교만족도': '학교생활만족도',
            '생활만족도': '학교생활만족도',
            '스마트폰': '스마트폰시간',
            '스마트폰 사용시간': '스마트폰시간',
            '공부': '공부시간',
            '수면': '수면시간'
        }
        raw_df = raw_df.rename(columns=column_rename_map)

        # 2. 데이터 미리보기
        st.header("2. 원본 데이터 미리보기")
        st.dataframe(raw_df.head())

        # 필수 컬럼 체크
        required_cols = ['학과', '학교생활만족도', '공부시간', '수면시간', '스마트폰시간']
        missing_cols = [c for c in required_cols if c not in raw_df.columns]

        if missing_cols:
            st.error(f"⚠️ CSV 파일에 필수 열(Column)이 누락되었습니다: {missing_cols}")
            st.info(f"현재 파일의 열 목록: {list(raw_df.columns)}")
            return

        # 3. 전처리 결과
        st.header("3. 데이터 전처리 결과")
        df, initial_rows, dropped_rows = preprocess_data(raw_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("총 입력 데이터", f"{initial_rows} 건")
        col2.metric("제외된 불량/이상치", f"{dropped_rows} 건")
        col3.metric("최종 분석 데이터", f"{len(df)} 건")

        if len(df) == 0:
            st.error("분석 가능한 유효 데이터가 없습니다.")
            return

        # 4. 학과 선택 필터
        st.header("4. 학과 선택 필터")
        department_list = ["전체 학과"] + sorted(list(df['학과'].dropna().unique()))
        selected_dept = st.selectbox("분석할 학과를 선택하세요:", department_list)

        if selected_dept == "전체 학과":
            filtered_df = df
        else:
            filtered_df = df[df['학과'] == selected_dept]

        avg_satisfaction = filtered_df['학교생활만족도'].mean()
        st.subheader(f"📌 [{selected_dept}] 평균 학교생활 만족도: **{avg_satisfaction:.2f} / 5.00 점**")

        # 5. 그래프 시각화
        st.header("5. 시각화 분석 결과")
        graph_col1, graph_col2 = st.columns(2)

        # [그래프 1] 막대그래프
        with graph_col1:
            st.subheader("📊 학과별 일일 생활습관 평균")
            dept_lifestyle = df.groupby('학과')[['공부시간', '수면시간', '스마트폰시간']].mean().reset_index()
            dept_lifestyle_melted = pd.melt(
                dept_lifestyle,
                id_vars=['학과'],
                value_vars=['공부시간', '수면시간', '스마트폰시간'],
                var_name='생활습관',
                value_name='평균시간'
            )

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=dept_lifestyle_melted, x='학과', y='평균시간', hue='생활습관', ax=ax1, palette="Set2")
            ax1.set_ylabel("시간 (Hours)")
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig1)
            plt.close(fig1)

        # [그래프 2] 히트맵
        with graph_col2:
            st.subheader(f"🔥 [{selected_dept}] 상관관계 히트맵")
            target_cols = ['공부시간', '수면시간', '스마트폰시간', '학교생활만족도']
            corr_matrix = filtered_df[target_cols].corr()

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

        # [그래프 3] 한줄 의견 단어 분석
        st.divider()
        st.subheader(f"💬 [{selected_dept}] 학생 한줄 의견 TOP 10 키워드 (긍정/부정)")

        if '한줄의견' in filtered_df.columns:
            word_df = analyze_opinion_words(filtered_df['한줄의견'])

            if not word_df.empty:
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                color_map = {"긍정": "#3498db", "부정": "#e74c3c", "기타": "#95a5a6"}

                sns.barplot(
                    data=word_df,
                    x='빈도수',
                    y='단어',
                    hue='구분',
                    dodge=False,
                    palette=color_map,
                    ax=ax3
                )
                ax3.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("분석할 단어 데이터가 충분하지 않습니다.")
        else:
            st.warning("업로드된 파일에 '한줄의견' 열이 없습니다.")

    else:
        st.info("💡 CSV 파일을 업로드하면 대시보드가 로드됩니다.")


if __name__ == "__main__":
    main()