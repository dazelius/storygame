import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict, Counter  # Counter 추가
import os
import openai
import networkx as nx
import json


# 페이지 설정
st.set_page_config(
    page_title="카카오톡 대화방 분석기",
    page_icon="🗒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 스타일 설정
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
    }
    .signal-card {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid #3d3d3d;
    }
    .high-signal {
        border-left: 5px solid #4CAF50;
    }
    .medium-signal {
        border-left: 5px solid #FFC107;
    }
    .low-signal {
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# 시그널 포인트 정의
SIGNAL_POINTS = {
    "빠른답장": {
        "weight": 10,
        "description": "1분 이내 답장",
        "threshold": 60  # seconds
    },
    "이모티콘": {
        "weight": 5,
        "description": "이모티콘 사용 빈도",
        "threshold": 0.2  # 메시지당 이모티콘 비율
    },
    "질문": {
        "weight": 8,
        "description": "질문 빈도",
        "threshold": 0.3  # 메시지당 질문 비율
    },
    "답장길이": {
        "weight": 7,
        "description": "평균 답장 길이",
        "threshold": 20  # 글자 수
    },
    "맞장구": {
        "weight": 6,
        "description": "맞장구 빈도",
        "threshold": 0.2  # 메시지당 맞장구 비율
    }
}

# 부정적 시그널 정의
NEGATIVE_SIGNALS = {
    "늦은답장": {
        "weight": -5,
        "description": "3시간 이상 답장 지연",
        "threshold": 10800  # seconds
    },
    "단답": {
        "weight": -3,
        "description": "5글자 이하 답장",
        "threshold": 5  # 글자 수
    },
    "화제전환": {
        "weight": -4,
        "description": "갑작스러운 화제 전환",
        "threshold": 0.3  # 화제 전환 비율
    }
}

def parse_kakao_chat(text: str) -> pd.DataFrame:
    """카카오톡 채팅 내용 파싱"""
    lines = text.split('\n')
    chat_data = []
    
    # 카톡 메시지 패턴: [이름] [시간] 메시지
    message_pattern = r'\[(.*?)\]\s\[(오전|오후)\s(\d{1,2}):(\d{2})\]\s(.*)'
    
    for line in lines:
        match = re.search(message_pattern, line)
        if match:
            name = match.group(1)
            am_pm = match.group(2)
            hour = int(match.group(3))
            minute = int(match.group(4))
            message = match.group(5)
            
            # 시간 변환
            if am_pm == "오후" and hour != 12:
                hour += 12
            elif am_pm == "오전" and hour == 12:
                hour = 0
                
            # 현재 날짜 사용 (실제로는 파일에서 날짜도 파싱해야 함)
            timestamp = datetime.now().replace(
                hour=hour, 
                minute=minute,
                second=0,
                microsecond=0
            )
            
            chat_data.append({
                'timestamp': timestamp,
                'name': name,
                'message': message
            })
    
    if not chat_data:
        st.error("채팅 데이터를 파싱할 수 없습니다. 올바른 카카오톡 대화 파일인지 확인해주세요.")
        return pd.DataFrame(columns=['timestamp', 'name', 'message'])
        
    return pd.DataFrame(chat_data)
    
def create_time_pattern(df: pd.DataFrame, target_names: list, my_name: str):
    """시간대별 대화 패턴 분석"""
    df['hour'] = df['timestamp'].dt.hour
    
    fig = go.Figure()
    
    # 색상 팔레트 정의
    colors = ['#ff4b6e', '#ff9eaf', '#ffb4c2', '#ffc9d3']  # 분홍계열
    
    # 각 대상자별 대화 패턴
    for idx, target_name in enumerate(target_names):
        target_counts = df[df['name'] == target_name].groupby('hour').size()
        # 없는 시간대는 0으로 채우기
        target_counts = target_counts.reindex(range(24), fill_value=0)
        
        fig.add_trace(go.Bar(
            x=target_counts.index,
            y=target_counts.values,
            name=f"{target_name}님의 대화",
            marker_color=colors[idx % len(colors)]  # 색상 순환
        ))
    
    # 내 대화 패턴
    my_counts = df[df['name'] == my_name].groupby('hour').size()
    # 없는 시간대는 0으로 채우기
    my_counts = my_counts.reindex(range(24), fill_value=0)
    
    fig.add_trace(go.Bar(
        x=my_counts.index,
        y=my_counts.values,
        name="나의 대화",
        marker_color='#4a90e2'  # 파란색
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="시간대별 대화 패턴 비교",
            font=dict(size=24, color='white')
        ),
        xaxis=dict(
            title="시간",
            ticktext=['오전 12시', '오전 3시', '오전 6시', '오전 9시', 
                     '오후 12시', '오후 3시', '오후 6시', '오후 9시'],
            tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
            tickangle=45,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title="메시지 수",
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white')
        ),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        ),
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    # x축 격자 추가
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def analyze_signals(df: pd.DataFrame, target_name: str) -> dict:
    """채팅 시그널 분석"""
    signals = {}
    
    # 타겟의 메시지만 필터링
    target_msgs = df[df['name'] == target_name]
    
    # 1. 답장 시간 분석
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    target_response_times = df[df['name'] == target_name]['time_diff'].dropna()
    
    signals['빠른답장'] = {
        'score': len(target_response_times[target_response_times < SIGNAL_POINTS['빠른답장']['threshold']]) / len(target_response_times) * 100,
        'detail': f"1분 이내 답장 비율: {len(target_response_times[target_response_times < 60]) / len(target_response_times):.1%}"
    }
    
    # 2. 이모티콘 사용 분석
    emoji_pattern = re.compile(r'[^\w\s,.]')
    emoji_count = target_msgs['message'].apply(lambda x: len(emoji_pattern.findall(x))).sum()
    signals['이모티콘'] = {
        'score': (emoji_count / len(target_msgs)) * 100,
        'detail': f"메시지당 이모티콘: {emoji_count / len(target_msgs):.1f}개"
    }
    
    # 3. 답장 길이 분석
    avg_length = target_msgs['message'].str.len().mean()
    signals['답장길이'] = {
        'score': min((avg_length / SIGNAL_POINTS['답장길이']['threshold']) * 100, 100),
        'detail': f"평균 답장 길이: {avg_length:.1f}자"
    }
    
    # 4. 질문 분석
    question_pattern = re.compile(r'[?？]')
    question_count = target_msgs['message'].apply(lambda x: bool(question_pattern.search(x))).sum()
    signals['질문'] = {
        'score': (question_count / len(target_msgs)) * 100,
        'detail': f"질문 비율: {question_count / len(target_msgs):.1%}"
    }
    
    return signals

def calculate_interest_score(signals: dict) -> float:
    """호감도 점수 계산"""
    total_weight = sum(SIGNAL_POINTS[k]['weight'] for k in signals.keys())
    weighted_score = sum(
        SIGNAL_POINTS[k]['weight'] * signals[k]['score'] 
        for k in signals.keys()
    )
    return min(weighted_score / total_weight, 100)

def create_time_pattern_plot(df: pd.DataFrame, target_name: str):
    """시간대별 대화 패턴 시각화"""
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df[df['name'] == target_name].groupby('hour').size()
    
    fig = go.Figure(data=go.Bar(
        x=hourly_counts.index,
        y=hourly_counts.values,
        marker_color='#ff4b6e'
    ))
    
    fig.update_layout(
        title="시간대별 대화 빈도",
        xaxis_title="시간",
        yaxis_title="메시지 수",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def calculate_avg_response_time(df: pd.DataFrame, target_name: str, my_name: str) -> float:
    """평균 답장 시간 계산 (분 단위)"""
    df = df.sort_values('timestamp')
    response_times = []
    
    for i in range(1, len(df)):
        if df.iloc[i]['name'] == target_name and df.iloc[i-1]['name'] == my_name:
            time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 60
            if time_diff < 60:  # 1시간 이내의 답장만 계산
                response_times.append(time_diff)
    
    return np.mean(response_times) if response_times else 0

def calculate_response_times(df: pd.DataFrame, target_name: str, my_name: str) -> list:
    """모든 답장 시간 리스트 반환 (초 단위)"""
    df = df.sort_values('timestamp')
    response_times = []
    
    for i in range(1, len(df)):
        if df.iloc[i]['name'] == target_name and df.iloc[i-1]['name'] == my_name:
            time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
            if time_diff < 3600:  # 1시간 이내의 답장만 포함
                response_times.append(time_diff)
    
    return response_times

def create_relationship_graph(relationships_data: dict):
    """향상된 AI 기반 관계 분석과 네트워크 그래프 시각화"""
    try:
        # 네트워크 그래프 생성
        G = nx.Graph()

        # 대화 강도에 따른 색상 맵 정의
        color_scale = px.colors.sequential.Viridis

        # 엣지 가중치와 노드 크기 계산을 위한 데이터 수집
        max_weight = 0
        node_interactions = defaultdict(int)
        edge_info = []

        # 관계 데이터를 기반으로 노드 및 엣지 추가
        for person1, connections in relationships_data.items():
            for person2, metrics in connections.items():
                # 상호작용 강도 계산
                interaction_strength = (
                    metrics['mentions'] * 2 +  # 직접 언급
                    metrics['time_overlap'] * 100 +  # 시간대 겹침
                    metrics.get('consecutive_talks', 0)  # 연속 대화
                )
                
                if interaction_strength > 0:
                    # 엣지 추가
                    G.add_edge(person1, person2, weight=interaction_strength)
                    edge_info.append((person1, person2, interaction_strength))
                    
                    # 노드별 총 상호작용 집계
                    node_interactions[person1] += interaction_strength
                    node_interactions[person2] += interaction_strength
                    
                    max_weight = max(max_weight, interaction_strength)

        if not G.edges():
            return go.Figure()

        # 노드 위치 계산 (spring_layout으로 더 자연스러운 배치)
        pos = nx.spring_layout(G, k=1, iterations=50)

        # 엣지 트레이스 생성
        edge_traces = []
        for person1, person2, weight in edge_info:
            x0, y0 = pos[person1]
            x1, y1 = pos[person2]
            
            # 상호작용 강도에 따른 선 두께와 색상 설정
            width = (weight / max_weight * 8) + 1  # 최소 1, 최대 9
            color_intensity = weight / max_weight
            color_idx = int(color_intensity * (len(color_scale) - 1))
            color = color_scale[color_idx]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=width,
                    color=color
                ),
                hoverinfo='text',
                text=f"{person1} ↔ {person2}<br>상호작용 강도: {weight:.1f}",
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # 노드 트레이스 생성
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 노드 크기는 총 상호작용량에 비례
            interaction_amount = node_interactions[node]
            size = np.sqrt(interaction_amount) * 2 + 20  # 기본 크기 20에 상호작용량 반영
            node_sizes.append(size)
            
            # 노드 색상은 상호작용 비율에 따라
            color_intensity = interaction_amount / max(node_interactions.values())
            color_idx = int(color_intensity * (len(color_scale) - 1))
            node_colors.append(color_scale[color_idx])
            
            # 호버 텍스트에 상세 정보 추가
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"총 상호작용: {interaction_amount:.0f}<br>"
            hover_text += "<b>주요 대화 상대:</b><br>"
            
            # 상위 3명의 대화 상대 추가
            connections = relationships_data.get(node, {})
            top_connections = sorted(
                connections.items(),
                key=lambda x: (
                    x[1]['mentions'] * 2 +
                    x[1]['time_overlap'] * 100 +
                    x[1].get('consecutive_talks', 0)
                ),
                reverse=True
            )[:3]
            
            for person, metrics in top_connections:
                interaction = (
                    metrics['mentions'] * 2 +
                    metrics['time_overlap'] * 100 +
                    metrics.get('consecutive_talks', 0)
                )
                hover_text += f"- {person}: {interaction:.0f}<br>"
            
            node_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )

        # Figure 생성
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=dict(
                    text='대화 참여자 관계도',
                    font=dict(size=24, color='white'),
                    y=0.95
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="💡 선이 굵고 진할수록 더 많은 대화를 나눈 사이입니다",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=14, color='white')
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        )

        # 컬러바 추가
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=max_weight,
                colorbar=dict(
                    title='상호작용 강도',
                    thickness=15,
                    title_font=dict(color='white'),
                    tickfont=dict(color='white'),
                    xanchor='left',
                    titleside='right'
                )
            ),
            hoverinfo='none'
        )
        fig.add_trace(colorbar_trace)

        return fig

    except Exception as e:
        st.error(f"관계도 생성 중 오류 발생: {str(e)}")
        return go.Figure()

def analyze_keywords(messages: pd.Series) -> list:
    """자주 사용된 키워드 분석"""
    # 불용어 정의
    ㄴstopwords = set(['그래서', '나는', '지금', 'https', 'www', 'naver', 'com', '샵검색', 'ㅇㅇ', '내가', '나도', '그런데', '하지만', '그리고', '그럼', '이제', '저기', '그게', '음', '아', '어', '응', '이모티콘', 'ㅋ', 'ㅋㅋ', 'ㅋㅋㅋ', 'ㅋㅋㅋㅋ', 'ㅎㅎ', 'ㄷㄷ', 'ㅎ','사진', '근데' , '일단' , '이제', '다들', '저거' ,'www', 'http', 'youtube', '삭제된 메시지입니다', '그리고', 
                        '네', '예', '아직', '우리', '많이', '존나', 'ㅋㅋㅋㅋㅋ', '저도', '같은데', '그냥', '너무', '진짜', '다시', '오늘', '보면' 'ㅋㅋㅋㅋㅋㅋ', 'ㅋㅋㅋㅋㅋㅋㅋ', '근데', '저기', '이거', '그거', '요', '은', '는', '이', '가', '을', '를', '에', '와', '과'])
        
    
    # 모든 메시지 합치기
    text = ' '.join(messages.dropna().astype(str))
    
    # 단어 분리 및 카운팅
    words = text.split()
    word_counts = defaultdict(int)
    
    for word in words:
        if len(word) > 1 and word not in stopwords:  # 2글자 이상, 불용어 제외
            word_counts[word] += 1
    
    # 상위 20개 단어 반환
    return sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:20]

def create_wordcloud(messages: pd.Series) -> plt.Figure:
    """워드클라우드 생성"""
    try:
        # 폰트 설치
        os.system('apt-get update && apt-get install fonts-nanum fonts-nanum-coding fonts-nanum-extra -y')
        
        # 폰트 경로 찾기
        font_dirs = [
            '/usr/share/fonts/truetype/nanum',
            '/usr/share/fonts',
            '~/.local/share/fonts',
            '/usr/local/share/fonts'
        ]
        
        font_path = None
        for font_dir in font_dirs:
            if os.path.exists(f"{font_dir}/NanumGothic.ttf"):
                font_path = f"{font_dir}/NanumGothic.ttf"
                break
        
        if not font_path:
            # 폰트 직접 다운로드
            os.system('wget https://raw.githubusercontent.com/apparition47/NanumGothic/master/NanumGothic.ttf -P /tmp/')
            font_path = '/tmp/NanumGothic.ttf'
        
        # matplotlib 폰트 설정
        plt.rcParams['font.family'] = 'NanumGothic'
        
        # 불용어 설정
        stopwords = set(['그래서', '나는', '지금', 'https', 'www', 'naver', 'com', '샵검색', 'ㅇㅇ', '내가', '나도', '그런데', '하지만', '그리고', '그럼', '이제', '저기', '그게', '음', '아', '어', '응', '이모티콘', 'ㅋ', 'ㅋㅋ', 'ㅋㅋㅋ', 'ㅋㅋㅋㅋ', 'ㅎㅎ', 'ㄷㄷ', 'ㅎ','사진', '근데' , '일단' , '이제', '다들', '저거' ,'www', 'http', 'youtube', '삭제된 메시지입니다', '그리고', 
                        '네', '예', '아직', '우리', '많이', '존나', 'ㅋㅋㅋㅋㅋ', '저도', '같은데', '그냥', '너무', '진짜', '다시', '오늘', '보면' 'ㅋㅋㅋㅋㅋㅋ', 'ㅋㅋㅋㅋㅋㅋㅋ', '근데', '저기', '이거', '그거', '요', '은', '는', '이', '가', '을', '를', '에', '와', '과'])
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            font_path=font_path,
            width=1200,
            height=600,
            background_color='black',
            colormap='Pastel1',
            prefer_horizontal=0.7,
            max_font_size=200,
            min_font_size=10,
            random_state=42,
            stopwords=stopwords,
            min_word_length=2,  # 최소 2글자 이상
            normalize_plurals=False,
            repeat=False
        )
        
        # 텍스트 전처리
        text = ' '.join(messages.dropna().astype(str))
        
        # 워드클라우드 생성
        wordcloud.generate(text)
        
        # 그림 생성
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
        
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류: {str(e)}")
        # 에러 발생시 기본 차트 반환
        fig = plt.figure(figsize=(15, 8))
        plt.text(0.5, 0.5, '워드클라우드를 생성할 수 없습니다\n한글 폰트 설치가 필요합니다', 
                ha='center', va='center')
        plt.axis('off')
        return fig

def analyze_topics(df: pd.DataFrame) -> dict:
    """주제 분석 (최적화 버전)"""
    # 간단한 키워드 기반 분석으로 대체
    keywords = {
        '일상': ['오늘', '내일', '어제', '밥', '먹', '잠', '집'],
        '감정': ['좋아', '싫어', '행복', '슬퍼', '화나', '웃'],
        '업무': ['일', '회사', '업무', '미팅', '프로젝트'],
        '취미': ['영화', '게임', '운동', '음악', '책'],
    }
    
    topic_counts = defaultdict(int)
    
    for msg in df['message']:
        if isinstance(msg, str):
            msg_lower = msg.lower()
            for topic, words in keywords.items():
                if any(word in msg_lower for word in words):
                    topic_counts[topic] += 1
    
    total = sum(topic_counts.values()) or 1
    return {topic: (count / total) * 100 for topic, count in topic_counts.items()}

def analyze_relationships(df: pd.DataFrame) -> dict:
    """향상된 관계 분석"""
    relationships = defaultdict(lambda: defaultdict(lambda: {
        'mentions': 0,  # 직접 언급
        'time_overlap': 0,  # 시간대 겹침
        'consecutive_talks': 0,  # 연속 대화
        'reaction_rate': 0,  # 반응률
        'common_topics': set()  # 공통 관심사
    }))
    
    # 멘션 분석 (이름 언급 횟수)
    names = set(df['name'].unique())
    for _, row in df.iterrows():
        for name in names:
            if name in str(row['message']) and name != row['name']:
                relationships[row['name']][name]['mentions'] += 1

    # 시간대 겹침 분석
    for name1 in names:
        for name2 in names:
            if name1 != name2:
                time_overlap = analyze_time_overlap(df, name1, name2)
                relationships[name1][name2]['time_overlap'] = time_overlap

    # 연속 대화 분석
    df_sorted = df.sort_values('timestamp')
    prev_name = None
    for name in df_sorted['name']:
        if prev_name and name != prev_name:
            relationships[prev_name][name]['consecutive_talks'] += 1
            relationships[name][prev_name]['consecutive_talks'] += 1
        prev_name = name

    # 반응률 분석 (상대방 메시지에 대한 반응 비율)
    for name1 in names:
        for name2 in names:
            if name1 != name2:
                reaction_rate = calculate_reaction_rate(df, name1, name2)
                relationships[name1][name2]['reaction_rate'] = reaction_rate

    # 공통 관심사 분석
    topic_patterns = {
        '게임': r'게임|플레이|캐릭터|아이템',
        '음식': r'맛있|먹|식당|카페',
        '영화/드라마': r'영화|드라마|배우|방영',
        '음악': r'노래|음악|가수|앨범',
        '운동': r'운동|헬스|요가|산책',
        '여행': r'여행|관광|숙소|항공',
        '일/공부': r'일|회사|공부|학교'
    }

    for name1 in names:
        person1_msgs = ' '.join(df[df['name'] == name1]['message'].astype(str))
        for name2 in names:
            if name1 != name2:
                person2_msgs = ' '.join(df[df['name'] == name2]['message'].astype(str))
                common_topics = set()
                
                for topic, pattern in topic_patterns.items():
                    if (re.search(pattern, person1_msgs, re.IGNORECASE) and 
                        re.search(pattern, person2_msgs, re.IGNORECASE)):
                        common_topics.add(topic)
                
                relationships[name1][name2]['common_topics'] = common_topics

    return relationships

def calculate_reaction_rate(df: pd.DataFrame, name1: str, name2: str) -> float:
    """두 사용자 간의 반응률 계산"""
    df = df.sort_values('timestamp')
    
    # 1분 이내의 반응을 "반응"으로 간주
    REACTION_THRESHOLD = 60  # seconds
    
    reactions = 0
    total_messages = 0
    
    for i in range(1, len(df)):
        if df.iloc[i-1]['name'] == name1 and df.iloc[i]['name'] == name2:
            time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
            if time_diff <= REACTION_THRESHOLD:
                reactions += 1
            total_messages += 1
    
    return reactions / max(total_messages, 1)


def analyze_chat_context(df: pd.DataFrame, target_names: list, my_name: str) -> dict:
    """대화 컨텍스트 분석 (향상된 버전)"""
    try:
        # 1. 데이터 샘플링 (최근 1000개 메시지만 분석)
        df_sample = df.sort_values('timestamp', ascending=False).head(1000)
        
        # 2. 기본 통계 계산
        stats = {
            'total_messages': len(df),
            'participants': list(df['name'].unique()),
            'date_range': (df['timestamp'].max() - df['timestamp'].min()).days,
            'topics': analyze_topics(df_sample),
            'relationships': analyze_relationships(df_sample)
        }
        
        # 3. 시간대별 대화량 분석
        hourly_stats = df_sample.groupby(df_sample['timestamp'].dt.hour).size()
        peak_hours = hourly_stats[hourly_stats > hourly_stats.mean()].index.tolist()
        
        # 4. 대화 참여도 분석
        participation = df_sample['name'].value_counts().to_dict()
        
        def get_gpt_analysis(text_sample, stats, peak_hours, participation):
            try:
                prompt = f"""
당신은 카카오톡 대화방을 심층 분석하는 전문가입니다. 
다음 대화 내용과 통계 데이터를 바탕으로 상세한 분석을 제공해주세요.

[기본 정보]
- 총 메시지 수: {stats['total_messages']:,}개
- 참여자 수: {len(stats['participants'])}명
- 분석 기간: {stats['date_range']}일
- 주요 활동 시간대: {', '.join(f'{h}시' for h in sorted(peak_hours))}

[분석할 대화 내용]
{text_sample}

다음 항목들을 상세히 분석해주세요:

1. 📋 주요 대화 주제 및 토픽
- 가장 많이 언급된 주제들
- 각 주제별 주요 키워드
- 대화의 전반적인 성향과 분위기

2. 💭 대화 내용 요약
- 주요 대화 흐름
- 중요한 논의사항이나 결정사항
- 인상적인 대화 포인트

3. 👥 대화 참여 패턴
- 각 참여자의 대화 스타일
- 대화 주도성과 반응성
- 특징적인 상호작용 패턴

4. 🌟 대화방의 특성
- 대화방의 전반적인 성격
- 주된 사용 목적과 용도
- 대화방만의 특별한 특징

각 항목에 대해 구체적인 예시와 함께 분석해주세요.
대화의 맥락을 잘 파악하여 의미있는 인사이트를 제공해주세요.
"""

                messages = [
                    {"role": "system", "content": "당신은 대화 분석 전문가로서, 객관적이고 통찰력 있는 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.7,
                    timeout=15
                )
                
                return response.choices[0].message['content']
            except Exception as e:
                st.warning(f"GPT 분석 중 오류 발생: {str(e)}")
                return "GPT 분석을 수행할 수 없습니다."
        
        # 5. 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analysis_steps = [
            '기본 통계 계산',
            '토픽 분석',
            '관계 분석',
            '대화 패턴 분석',
            'GPT 분석'
        ]
        
        for i, step in enumerate(analysis_steps):
            status_text.text(f'분석 중... {step}')
            progress_bar.progress((i + 1) * 20)
            time.sleep(0.1)
        
        # 6. 대화 샘플 준비 (시간순으로 정렬된 최근 100개 메시지)
        recent_messages = df_sample.sort_values('timestamp').tail(100)
        chat_sample = []
        
        # 대화 맥락을 유지하며 샘플 구성
        prev_date = None
        for _, msg in recent_messages.iterrows():
            curr_date = msg['timestamp'].strftime('%Y-%m-%d')
            if curr_date != prev_date:
                chat_sample.append(f"\n[{curr_date}]\n")
                prev_date = curr_date
            chat_sample.append(f"{msg['name']}: {msg['message']}")
        
        chat_text = '\n'.join(chat_sample)
        
        # 7. 최종 결과 반환
        analysis_result = {
            **stats,
            'peak_hours': peak_hours,
            'participation': participation,
            'gpt_analysis': get_gpt_analysis(chat_text, stats, peak_hours, participation)
        }
        
        progress_bar.progress(100)
        status_text.text('분석 완료!')
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return analysis_result
        
    except Exception as e:
        st.error(f"분석 중 오류 발생: {str(e)}")
        return None



def analyze_relationships(df: pd.DataFrame) -> dict:
    """대화 참여자들 간의 관계 분석"""
    relationships = {}
    
    # 각 참여자별 상호작용 분석
    for name1 in df['name'].unique():
        relationships[name1] = {}
        for name2 in df['name'].unique():
            if name1 != name2:
                # 연속 대화 횟수
                consecutive_talks = 0
                # 멘션 횟수
                mentions = len(df[
                    (df['name'] == name1) & 
                    (df['message'].str.contains(name2, na=False))
                ])
                # 대화 시간대 겹침
                time_overlap = analyze_time_overlap(df, name1, name2)
                
                relationships[name1][name2] = {
                    'mentions': mentions,
                    'time_overlap': time_overlap,
                    'consecutive_talks': consecutive_talks
                }
    
    return relationships

def generate_suggestions(analysis: dict) -> dict:
    """대화 분석 기반으로 개선 제안 생성"""
    # 기본 제안 정의
    default_suggestions = {
        'positive': [
            "대화가 꾸준히 이어지고 있습니다.",
            "기본적인 예의를 지키며 대화하고 있습니다.",
            "서로의 의견을 존중하며 대화하고 있습니다."
        ],
        'improvements': [
            "더 자주 대화를 나눠보세요.",
            "다양한 주제로 대화를 확장해보세요.",
            "상대방의 이야기에 더 적극적으로 반응해보세요."
        ]
    }

    try:
        if not analysis:  # analysis가 None이거나 비어있는 경우
            return default_suggestions

        suggestions = {
            'positive': [],
            'improvements': []
        }
        
        # 대화량 분석
        if 'participants' in analysis and len(analysis['participants']) > 2:
            suggestions['positive'].append("다양한 참여자들과 활발한 대화가 이루어지고 있습니다.")
        
        # 주제 다양성 분석
        if 'topics' in analysis and analysis['topics']:
            suggestions['positive'].append("다양한 주제로 대화가 진행되어 대화가 풍부합니다.")
        
        # 기본 긍정적 피드백 추가
        suggestions['positive'].extend([
            "정기적으로 대화가 이어지고 있습니다.",
            "서로 존중하는 대화가 이루어지고 있습니다.",
            "적절한 이모티콘 사용으로 감정 전달이 잘 되고 있습니다."
        ])
        
        # 개선 제안 추가
        suggestions['improvements'].extend([
            "더 많은 질문으로 상대방의 이야기를 이끌어내보세요.",
            "긴 대화 공백이 있을 때는 간단한 인사로 대화를 이어가보세요.",
            "상대방의 관심사에 대해 더 깊이 있는 대화를 나눠보세요.",
            "이모티콘과 함께 구체적인 감정 표현을 해보세요."
        ])
        
        return suggestions if suggestions['positive'] or suggestions['improvements'] else default_suggestions
        
    except Exception as e:
        st.error(f"제안 생성 중 오류 발생: {str(e)}")
        return default_suggestions

def display_suggestions(analysis: dict):
    """AI 제안 표시"""
    st.markdown("## 💡 AI의 제안")
    suggestions = generate_suggestions(analysis)
    
    if not suggestions:  # suggestions가 None인 경우 처리
        suggestions = {
            'positive': ["대화 분석에 기반한 제안을 생성할 수 없습니다."],
            'improvements': ["대화 분석에 기반한 개선점을 생성할 수 없습니다."]
        }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✨ 긍정적인 포인트")
        for point in suggestions['positive']:
            st.success(point)

    with col2:
        st.markdown("### 🎯 개선 포인트")
        for point in suggestions['improvements']:
            st.warning(point)

def analyze_time_overlap(df: pd.DataFrame, name1: str, name2: str) -> float:
    """두 참여자의 대화 시간대 겹침 정도 분석"""
    person1_times = df[df['name'] == name1]['timestamp'].dt.hour.value_counts()
    person2_times = df[df['name'] == name2]['timestamp'].dt.hour.value_counts()
    
    overlap = sum(min(person1_times.get(hour, 0), person2_times.get(hour, 0)) 
                 for hour in range(24))
    
    total = sum(max(person1_times.get(hour, 0), person2_times.get(hour, 0)) 
                for hour in range(24))
    
    return overlap / total if total > 0 else 0

def find_highlight_messages(df: pd.DataFrame, target_names: list, my_name: str) -> dict:
    """인상적인 대화 찾기"""
    try:
        highlights = {
            'emotional_messages': [],
            'discussion_messages': [],
            'quick_responses': []
        }
        
        # 감정 표현이 포함된 메시지 찾기
        emotion_patterns = [
            r'[ㅋㅎ]{2,}',  # 웃음
            r'[ㅠㅜ]{2,}',  # 슬픔
            r'[!?]{2,}',   # 강한 감정
            r'😊|😄|😢|😭|😡|❤️|👍|🙏'  # 이모티콘
        ]
        
        for target_name in target_names:
            target_msgs = df[df['name'] == target_name].copy()
            
            # 1. 감정이 풍부한 메시지
            for pattern in emotion_patterns:
                emotional = target_msgs[target_msgs['message'].str.contains(pattern, regex=True, na=False)]
                for _, msg in emotional.iterrows():
                    highlights['emotional_messages'].append({
                        'timestamp': msg['timestamp'],
                        'name': msg['name'],
                        'message': msg['message']
                    })
            
            # 2. 긴 메시지 (활발한 토론)
            long_messages = target_msgs[target_msgs['message'].str.len() > 50]
            for _, msg in long_messages.iterrows():
                highlights['discussion_messages'].append({
                    'timestamp': msg['timestamp'],
                    'name': msg['name'],
                    'message': msg['message']
                })
            
            # 3. 빠른 답장
            target_msgs['prev_msg_time'] = target_msgs['timestamp'].shift(1)
            target_msgs['response_time'] = (target_msgs['timestamp'] - target_msgs['prev_msg_time']).dt.total_seconds()
            
            quick_responses = target_msgs[
                (target_msgs['response_time'] < 60) &  # 1분 이내 답장
                (target_msgs['response_time'] > 0)     # 음수 제외
            ]
            
            for _, msg in quick_responses.iterrows():
                highlights['quick_responses'].append({
                    'timestamp': msg['timestamp'],
                    'name': msg['name'],
                    'message': msg['message']
                })
        
        # 각 카테고리별로 최대 5개까지만 선택
        for category in highlights:
            highlights[category] = sorted(
                highlights[category],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:5]
        
        return highlights
        
    except Exception as e:
        st.error(f"대화 하이라이트 분석 중 오류: {str(e)}")
        return {
            'emotional_messages': [],
            'discussion_messages': [],
            'quick_responses': []
        }

def analyze_conversation_stats(df: pd.DataFrame) -> dict:
    """참여자별 대화량 분석"""
    conversation_stats = df.groupby('name').size().to_dict()
    return conversation_stats

def create_conversation_chart(conversation_stats: dict) -> go.Figure:
    """참여자별 대화량 시각화"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(conversation_stats.keys()),
            y=list(conversation_stats.values()),
            marker_color='#4a90e2'
        )
    ])
    fig.update_layout(
        title="참여자별 대화량",
        xaxis_title="참여자",
        yaxis_title="메시지 수",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def analyze_emotions(df: pd.DataFrame) -> dict:
    """감정 분석"""
    # 감정 분석 로직 구현
    pass

def create_emotion_wordcloud(df: pd.DataFrame) -> plt.Figure:
    """감정별 키워드 워드클라우드 생성"""
    # 감정별 키워드 워드클라우드 생성 로직 구현
    pass

def generate_suggestions(analysis: dict) -> dict:
    """개선 제안 생성"""
    # 개선 제안 생성 로직 구현
    pass

def create_emotion_chart(emotion_stats: dict) -> go.Figure:
    """감정 분석 결과를 시각화하는 차트 생성"""
    try:
        # 기본 감정이 없는 경우 샘플 데이터 사용
        if not emotion_stats:
            emotion_stats = {
                "긍정": 0,
                "중립": 0,
                "부정": 0
            }
        
        # 감정별 색상 매핑
        color_map = {
            "긍정": "#4CAF50",  # 초록
            "중립": "#2196F3",  # 파랑
            "부정": "#F44336",  # 빨강
            "기쁨": "#8BC34A",  # 연한 초록
            "슬픔": "#9C27B0",  # 보라
            "화남": "#FF5722",  # 주황
            "놀람": "#FFEB3B",  # 노랑
        }
        
        # 데이터 정렬
        sorted_emotions = dict(sorted(emotion_stats.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
        
        # 차트 생성
        fig = go.Figure()
        
        # 바 차트 추가
        fig.add_trace(go.Bar(
            x=list(sorted_emotions.keys()),
            y=list(sorted_emotions.values()),
            marker_color=[color_map.get(emotion, "#757575") for emotion in sorted_emotions.keys()],
            text=[f"{value:.1f}%" for value in sorted_emotions.values()],
            textposition='auto',
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text="대화 감정 분석",
                font=dict(size=24, color='white'),
                y=0.95
            ),
            xaxis=dict(
                title="감정",
                tickfont=dict(color='white'),
                showgrid=False
            ),
            yaxis=dict(
                title="비율 (%)",
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=50),
            bargap=0.3
        )
        
        # 호버 템플릿 설정
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "비율: %{y:.1f}%<br>" +
                         "<extra></extra>"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"감정 차트 생성 중 오류 발생: {str(e)}")
        return go.Figure()

def analyze_emotions(df: pd.DataFrame) -> dict:
    """대화 내용의 감정 분석"""
    try:
        # 감정 키워드 정의
        emotion_keywords = {
            "긍정": ["좋아", "감사", "행복", "기쁘", "신나", "최고", "사랑", "웃", "ㅋㅋ", "ㅎㅎ", "😊", "😄", "👍"],
            "부정": ["싫어", "짜증", "화나", "슬프", "힘들", "어려", "나빠", "ㅠㅠ", "ㅜㅜ", "😢", "😭", "😡"],
            "중립": ["그래", "음", "아", "네", "응", "글쎄", "그렇", "아하", "흠"],
        }
        
        # 전체 메시지 수
        total_messages = len(df)
        emotion_counts = {emotion: 0 for emotion in emotion_keywords.keys()}
        
        # 각 메시지의 감정 분석
        for message in df['message']:
            if isinstance(message, str):  # 문자열인 경우만 처리
                message = message.lower()  # 소문자 변환
                for emotion, keywords in emotion_keywords.items():
                    if any(keyword in message for keyword in keywords):
                        emotion_counts[emotion] += 1
        
        # 비율 계산
        emotion_percentages = {
            emotion: (count / total_messages) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return emotion_percentages
        
    except Exception as e:
        st.error(f"감정 분석 중 오류 발생: {str(e)}")
        return {}

def analyze_chat_topics(messages: pd.Series) -> dict:
    """대화 주제 분석"""
    topics = {
        '일상': ['밥', '먹', '잠', '피곤', '영화', '드라마', '주말'],
        '감정': ['좋아', '싫', '슬프', '행복', '웃', '힘들'],
        '업무': ['회사', '일', '업무', '프로젝트', '미팅'],
        '취미': ['게임', '운동', '음악', '독서', '영화'],
    }
    
    topic_counts = defaultdict(int)
    text = ' '.join(messages.dropna().astype(str))
    
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in text:
                topic_counts[topic] += text.count(keyword)
    
    return dict(topic_counts)

def create_topic_chart(topics_data: dict) -> go.Figure:
    """주제별 대화 분포를 시각화하는 차트 생성"""
    try:
        # 데이터 정렬 (값이 큰 순서대로)
        sorted_topics = dict(sorted(topics_data.items(), key=lambda x: x[1], reverse=True))
        
        # 주제별 색상 매핑
        color_map = {
            '일상': '#FF9999',  # 분홍빛 레드
            '감정': '#66B2FF',  # 하늘색
            '업무': '#99FF99',  # 연한 초록
            '취미': '#FFCC99',  # 연한 주황
            '기타': '#CC99FF'   # 연한 보라
        }
        
        colors = [color_map.get(topic, '#CCCCCC') for topic in sorted_topics.keys()]
        
        # 파이 차트 생성
        fig = go.Figure(data=[
            go.Pie(
                labels=list(sorted_topics.keys()),
                values=list(sorted_topics.values()),
                hole=0.4,  # 도넛 차트 스타일
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(color='white'),
                hovertemplate="<b>%{label}</b><br>" +
                            "메시지 수: %{value}<br>" +
                            "비율: %{percent}<br>" +
                            "<extra></extra>"
            )
        ])
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text="주제별 대화 분포",
                font=dict(size=24, color='white'),
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0)',
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    text="대화<br>주제",
                    x=0.5,
                    y=0.5,
                    font=dict(size=20, color='white'),
                    showarrow=False
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        st.error(f"토픽 차트 생성 중 오류 발생: {str(e)}")
        # 오류 발생 시 빈 Figure 반환
        return go.Figure()

def create_detailed_wordcloud(messages: pd.Series) -> plt.Figure:
    """감정별 색상이 다른 워드클라우드 생성"""
    try:
        text = ' '.join(messages.dropna().astype(str))
        
        # 감정별 색상 매핑
        color_func = lambda word, font_size, position, orientation, random_state=None, **kwargs: (
            '#ff6b6b' if word in ['좋아', '행복', '웃'] else  # 긍정
            '#4ecdc4' if word in ['화이팅', '응원', '파이팅'] else  # 격려
            '#95a5a6' if word in ['그래', '음', '아'] else  # 중립
            'white'  # 기본
        )
        
        # 리눅스 기본 폰트 경로 시도
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 나눔고딕
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # DejaVu Sans
            None  # 기본 폰트
        ]
        
        # 사용 가능한 폰트 찾기
        font_path = None
        for path in font_paths:
            if path and os.path.exists(path):
                font_path = path
                break
        
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='black',
            color_func=color_func,
            font_path=font_path,  # 찾은 폰트 사용
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=100,
            random_state=42
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
        
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류 발생: {str(e)}")
        return None

# 새로운 분석 함수들
def get_favorite_emojis(messages: pd.Series, top_k: int = 3) -> list:
    """자주 사용하는 이모티콘 분석"""
    emoji_pattern = re.compile(r'[😀-🙏🌀-🗿]+|[\u2600-\u26FF\u2700-\u27BF]')
    emojis = []
    for msg in messages:
        if isinstance(msg, str):
            emojis.extend(emoji_pattern.findall(msg))
    return Counter(emojis).most_common(top_k)

def get_frequent_words(messages: pd.Series, top_k: int = 5) -> list:
    """자주 사용하는 단어 분석 (불용어 제외)"""
    stopwords = set(['그래서', '나는', '지금', '그런데', '그리고', '그럼', '네', '예', '음', '아'])
    words = []
    for msg in messages:
        if isinstance(msg, str):
            words.extend([w for w in msg.split() if len(w) > 1 and w not in stopwords])
    return Counter(words).most_common(top_k)

def calculate_conversation_starter_ratio(df: pd.DataFrame, name: str) -> float:
    """대화 시작 비율 계산"""
    df = df.sort_values('timestamp')
    conversation_gaps = df['timestamp'].diff() > pd.Timedelta(minutes=30)
    conversation_starts = df[conversation_gaps]['name'] == name
    return round(conversation_starts.sum() / conversation_gaps.sum() * 100, 1)

def analyze_emotion_patterns(messages: pd.Series) -> dict:
    """감정 표현 패턴 분석"""
    patterns = {
        '긍정': r'[ㅋㅎ]{2,}|😊|😄|😆|❤️|👍|좋아|감사|행복',
        '부정': r'[ㅠㅜ]{2,}|😢|😭|😡|😱|슬퍼|힘들|짜증',
        '놀람': r'[!?]{2,}|😮|😲|헐|대박|미쳤|실화',
        '애정': r'❤️|🥰|😘|💕|사랑|보고싶|그리워'
    }
    
    emotion_counts = {}
    for emotion, pattern in patterns.items():
        count = sum(1 for msg in messages if isinstance(msg, str) and re.search(pattern, msg))
        if count > 0:
            emotion_counts[emotion] = count
            
    total = sum(emotion_counts.values()) or 1
    return {k: round(v/total * 100, 1) for k, v in emotion_counts.items()}

def analyze_conversation_leadership(df: pd.DataFrame, name: str) -> float:
    """대화 주도성 분석"""
    user_msgs = df[df['name'] == name]
    total_msgs = len(df)

    starter_ratio = calculate_conversation_starter_ratio(df, name)
    msg_ratio = len(user_msgs) / total_msgs * 100
    question_pattern = r'[?？]'
    question_ratio = sum(user_msgs['message'].str.contains(question_pattern, na=False)) / len(user_msgs) * 100
    
    leadership_score = (starter_ratio + msg_ratio + question_ratio) / 3
    return round(leadership_score, 1)

def analyze_humor_patterns(messages: pd.Series) -> str:
    """유머 사용 패턴 분석"""
    humor_patterns = {
        '이모티콘 유머': r'[ㅋㅎ]{3,}|😆|🤣',
        '드립': r'드립|개그|농담|장난',
        '재치있는 표현': r'웃긴|재밌|웃음|재치'
    }
    
    humor_counts = {k: sum(messages.str.contains(v, na=False)) for k, v in humor_patterns.items()}
    total_msgs = len(messages)
    
    if sum(humor_counts.values()) / total_msgs < 0.1:
        return "유머 사용 적음"
    
    main_humor = max(humor_counts.items(), key=lambda x: x[1])
    return f"{main_humor[0]} 위주의 유머 사용 ({main_humor[1]}회)"

def get_reaction_patterns(df: pd.DataFrame, name: str) -> str:
    """반응 패턴 분석"""
    user_responses = df[df['name'] == name]
    quick_responses = sum(df['timestamp'].diff().dt.total_seconds() < 60)
    
    if len(user_responses) == 0:
        return "반응 패턴 분석 불가"
    
    patterns = []
    if quick_responses / len(user_responses) > 0.3:
        patterns.append("빠른 반응")
    if sum(user_responses['message'].str.contains(r'[ㅋㅎ]{2,}|[!?]{2,}', na=False)) / len(user_responses) > 0.3:
        patterns.append("감정적 반응")
    if sum(user_responses['message'].str.contains(r'그래요?|정말요?|진짜요?', na=False)) / len(user_responses) > 0.2:
        patterns.append("공감적 반응")
        
    return ", ".join(patterns) if patterns else "일반적인 반응"

def analyze_link_sharing(messages: pd.Series) -> str:
    """링크 공유 성향 분석"""
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    link_count = sum(messages.str.contains(link_pattern, na=False))
    
    if link_count == 0:
        return "링크 공유 없음"
    elif link_count < 5:
        return f"가끔 링크 공유 ({link_count}회)"
    else:
        return f"활발한 정보 공유 ({link_count}회)"

def analyze_question_patterns(messages: pd.Series) -> str:
    """질문 패턴 분석"""
    question_types = {
        '일반 질문': r'\?|궁금|어때|할까',
        '의견 요청': r'어떻게|어떨까|괜찮을까|좋을까',
        '정보 요청': r'뭐|언제|어디|누구|얼마'
    }
    
    type_counts = {k: sum(messages.str.contains(v, na=False)) for k, v in question_types.items()}
    total = sum(type_counts.values())
    
    if total == 0:
        return "질문 적음"
    
    main_type = max(type_counts.items(), key=lambda x: x[1])
    return f"{main_type[0]} 위주 ({main_type[1]}회)"

def analyze_personality_with_gpt(df: pd.DataFrame, name: str) -> dict:
    """GPT를 활용한 사용자 성격 분석"""
    try:
        # 해당 사용자의 메시지만 추출
        user_messages = df[df['name'] == name]['message'].tolist()
        
        # 분석을 위한 메시지 샘플링 (최근 100개)
        sample_size = min(100, len(user_messages))
        message_sample = user_messages[-sample_size:]

        # 감정 표현 분석
        emotion_pattern = re.compile(r'[ㅋㅎ]{2,}|[ㅠㅜ]{2,}|[!?]{2,}|😊|😄|😢|😭|😡|❤️|👍|🙏')
        emotion_count = sum(1 for msg in message_sample if isinstance(msg, str) and emotion_pattern.search(msg))
        emotion_ratio = emotion_count / sample_size if sample_size > 0 else 0

        # 이모티콘 사용 분석
        emoji_pattern = re.compile(r'[😀-🙏🌀-🗿]+|[\u2600-\u26FF\u2700-\u27BF]|\[이모티콘\]')
        emoji_count = sum(len(emoji_pattern.findall(str(msg))) for msg in message_sample)
        emoji_ratio = emoji_count / sample_size if sample_size > 0 else 0

        # 자주 사용하는 단어 분석
        words = []
        stopwords = {'그래서', '나는', '지금', '그런데', '그리고', '그럼', '네', '예', '음', '아', 
                    '저도', '근데', '저는', '제가', '좀', '이제', '그냥', '진짜', '아니', '그건'}
        for msg in message_sample:
            if isinstance(msg, str):
                words.extend([word for word in msg.split() if len(word) > 1 and word not in stopwords])
        word_counter = Counter(words)
        frequent_words = word_counter.most_common(5)

        # 질문 패턴 분석
        question_pattern = re.compile(r'[?？]|어때|할까|뭐|언제|어디|누구')
        question_count = sum(1 for msg in message_sample if isinstance(msg, str) and question_pattern.search(msg))
        question_ratio = question_count / sample_size if sample_size > 0 else 0

        # 대화 시작 비율 분석
        df_sorted = df.sort_values('timestamp')
        conversation_gaps = df_sorted['timestamp'].diff() > pd.Timedelta(minutes=30)
        conversation_starts = df_sorted[conversation_gaps]
        starter_count = sum(conversation_starts['name'] == name)
        starter_ratio = starter_count / len(conversation_starts) if len(conversation_starts) > 0 else 0

        # 메시지 길이 분석
        msg_lengths = [len(str(msg)) for msg in message_sample]
        avg_length = sum(msg_lengths) / len(msg_lengths) if msg_lengths else 0
        
        # GPT 프롬프트 구성
        metrics = {
            "감정표현비율": round(emotion_ratio * 100, 1),
            "이모티콘비율": round(emoji_ratio * 100, 1),
            "질문비율": round(question_ratio * 100, 1),
            "대화시작비율": round(starter_ratio * 100, 1),
            "평균메시지길이": round(avg_length, 1),
            "자주쓰는단어": frequent_words
        }

        prompt = f"""
당신은 20년 경력의 심리학 박사이자 대화 분석 전문가입니다. {name}님의 카카오톡 대화 데이터를 기반으로 심층적인 성격 분석을 진행해주세요.

[분석 데이터]
1. 대화 패턴:
- 대화 시작 비율: {metrics['대화시작비율']}% (높을수록 대화 주도적)
- 질문 비율: {metrics['질문비율']}%
- 평균 메시지 길이: {metrics['평균메시지길이']}자
- 감정 표현 비율: {metrics['감정표현비율']}%
- 이모티콘 사용 비율: {metrics['이모티콘비율']}%

2. 자주 사용하는 단어 (상위 5개):
{', '.join(f'{word}({count}회)' for word, count in metrics['자주쓰는단어'])}

[심층 분석 요청사항]
1. 🎯 핵심 성격 특성 (구체적 근거 필수)
- 대화 데이터에서 발견되는 가장 두드러진 성격적 특징 3가지를 구체적인 수치와 예시와 함께 설명해주세요
- 각 특징이 대화에서 어떻게 구체적으로 드러나는지 실제 사용 패턴을 바탕으로 설명해주세요
- 이 사람만의 독특한 매력 포인트를 대화 스타일에서 발견되는 특별한 점과 연결지어 설명해주세요

2. 🗣️ 의사소통 스타일 분석
- 대화를 이끌어가는 특별한 방식이나 패턴
- 감정과 생각을 표현할 때의 독특한 특징
- 갈등이나 긴장 상황에서의 대처 방식
- 유머나 위트의 사용 패턴과 그 효과

3. 💝 관계 형성 방식
- 친밀감을 표현하는 고유한 방법
- 타인을 배려하거나 지지하는 특별한 패턴
- 그룹 내에서의 역할과 영향력
- 관계 유지에 있어서의 강점

4. 🎭 MBTI 성향 추정
- 외향성/내향성 (E/I): {metrics['대화시작비율']}%의 대화 시작 비율 등 참고
- 감각/직관 (S/N): 구체적 표현과 추상적 표현의 비율 참고
- 사고/감정 (T/F): {metrics['감정표현비율']}%의 감정 표현 비율 등 참고
- 판단/인식 (J/P): 대화 패턴과 응답 스타일 참고

5. 💡 잠재력과 발전 포인트
- 현재 가장 잘 발휘되고 있는 강점
- 더 개발하면 좋을 잠재적 재능
- 대인관계에서의 특별한 영향력

분석을 통해 {name}님의 진정한 매력과 특별한 가치가 잘 드러나도록 심층적이고 구체적인 분석을 부탁드립니다.
"""

        messages = [
            {"role": "system", "content": "당신은 심리학자이자 성격 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
            timeout=15
        )

        return {
            "name": name,
            "metrics": metrics,
            "gpt_analysis": response.choices[0].message['content']
        }
        
    except Exception as e:
        st.error(f"성격 분석 중 오류 발생: {str(e)}")
        return None


def calculate_personality_metrics(message_text: str) -> dict:
    """성격 특성 점수 계산"""
    # 각 특성별 패턴 정의
    metrics_patterns = {
        "매력도": {
            "patterns": [
                r'ㅋㅋ|ㅎㅎ|웃긴|재미|신기|멋지',
                r'센스|배려|친절|상냥|다정|착하',
                r'😊|🤣|😂|😍|🥰|😘|😅|❤|[이모티콘]'
            ],
            "weight": 1.2
        },
        "친화력": {
            "patterns": [
                r'같이|우리|함께|저희|모두|다같이',
                r'고마워|감사|죄송|미안|부탁|도와',
                r'맞아|그래|응|네|당연|그렇지'
            ],
            "weight": 1.1
        },
        "활발도": {
            "patterns": [
                r'하자|가자|놀자|먹자|보자|할까',
                r'신나|재밌|즐거|행복|좋아|대박',
                r'\!+|\?+|ㅋ+|ㅎ+|~+'
            ],
            "weight": 1.0
        },
        "감성력": {
            "patterns": [
                r'좋아|행복|그립|보고싶|사랑|설레',
                r'아름|예쁘|귀엽|멋지|근사|대단',
                r'ㅠㅠ|ㅜㅜ|😢|😭|💕|❤'
            ],
            "weight": 1.0
        },
        "지적호기심": {
            "patterns": [
                r'왜|어떻게|무엇|언제|어디|누구',
                r'관심|궁금|알고|싶|찾|공부',
                r'정보|지식|이해|학습|배우|연구'
            ],
            "weight": 1.1
        }
    }

    scores = {}
    for metric, info in metrics_patterns.items():
        metric_score = 0
        patterns = info["patterns"]
        weight = info["weight"]
        
        for pattern_group in patterns:
            try:
                matches = len(re.findall(pattern_group, message_text, re.IGNORECASE))
                metric_score += min(100, matches * 5)
            except Exception as e:
                print(f"패턴 매칭 오류: {pattern_group} - {str(e)}")
                continue
        
        # 패턴 그룹 수로 나누어 평균 계산
        avg_score = (metric_score / len(patterns)) * weight
        scores[metric] = round(min(100, max(0, avg_score)), 1)
    
    return scores

def analyze_mbti_patterns(messages: pd.Series) -> dict:
    """MBTI 관련 패턴 분석"""
    mbti_indicators = {
        "E": {
            "patterns": [
                r'같이|우리|놀자|만나',
                r'재미있|신나|즐거|파티',
                r'사람들|친구|모임|약속'
            ],
            "counter_patterns": [
                r'혼자|집에|쉬고|조용',
                r'피곤|지친|힘들|귀찮',
                r'개인|독립|자유|여유'
            ]
        },
        "I": {
            "patterns": [
                r'혼자|집|책|음악',
                r'조용|평화|쉬고|여유',
                r'생각|고민|느낌|마음'
            ],
            "counter_patterns": [
                r'파티|놀자|모임|같이',
                r'사람들|친구들|우리|다같이',
                r'시끌|북적|왁자|떠들'
            ]
        },
        "S": {
            "patterns": [
                r'지금|여기|오늘|내일',
                r'실제|현실|경험|사실',
                r'구체|정확|확실|직접'
            ],
            "counter_patterns": [
                r'상상|미래|가능성|예측',
                r'아이디어|영감|직감|느낌',
                r'의미|상징|철학|관계'
            ]
        },
        "N": {
            "patterns": [
                r'상상|아이디어|영감|가능',
                r'의미|이유|원리|이론',
                r'미래|변화|혁신|창의'
            ],
            "counter_patterns": [
                r'현실|사실|경험|직접',
                r'구체|정확|확실|지금',
                r'여기|오늘|내일|실제'
            ]
        },
        "T": {
            "patterns": [
                r'논리|이성|분석|판단',
                r'객관|정확|효율|성과',
                r'해결|원인|결과|방법'
            ],
            "counter_patterns": [
                r'감정|느낌|마음|공감',
                r'위로|격려|지지|응원',
                r'좋아|싫어|행복|슬퍼'
            ]
        },
        "F": {
            "patterns": [
                r'감정|느낌|마음|공감',
                r'위로|격려|지지|응원',
                r'사랑|행복|좋아|그리워'
            ],
            "counter_patterns": [
                r'논리|이성|분석|판단',
                r'객관|정확|효율|성과',
                r'해결|원인|결과|방법'
            ]
        },
        "J": {
            "patterns": [
                r'계획|일정|약속|규칙',
                r'정리|체계|순서|단계',
                r'결정|확실|마감|완료'
            ],
            "counter_patterns": [
                r'갑자기|즉흥|자유|융통',
                r'때되면|나중|미루|언젠가',
                r'변경|수정|유연|적응'
            ]
        },
        "P": {
            "patterns": [
                r'자유|즉흥|유연|변화',
                r'가능성|선택|대안|융통',
                r'편한|느긋|여유|자연'
            ],
            "counter_patterns": [
                r'계획|일정|약속|규칙',
                r'정리|체계|순서|단계',
                r'결정|확실|마감|완료'
            ]
        }
    }

    # 각 지표별 점수 계산
    msg_text = ' '.join(messages.astype(str))
    mbti_scores = {}
    
    for indicator, patterns in mbti_indicators.items():
        positive_score = sum(len(re.findall(p, msg_text, re.IGNORECASE)) for p in patterns["patterns"])
        negative_score = sum(len(re.findall(p, msg_text, re.IGNORECASE)) for p in patterns["counter_patterns"])
        total_score = positive_score - negative_score
        mbti_scores[indicator] = round(min(100, max(0, 50 + (total_score * 5))), 1)

    # 각 지표쌍의 강한 쪽 선택
    trait_pairs = [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]
    predicted_mbti = ''
    probabilities = {}
    
    for pair in trait_pairs:
        score1, score2 = mbti_scores[pair[0]], mbti_scores[pair[1]]
        stronger = pair[0] if score1 > score2 else pair[1]
        confidence = abs(score1 - score2) / 100  # 0~1 사이 값
        predicted_mbti += stronger
        probabilities[f"{pair[0]}/{pair[1]}"] = round(max(score1, score2), 1)

    return {
        "predicted_type": predicted_mbti,
        "confidence_scores": probabilities,
        "detailed_scores": mbti_scores
    }

def analyze_interests(messages: pd.Series) -> dict:
    """관심사 분석"""
    interest_categories = {
        "엔터테인먼트": {
            "patterns": [
                r'영화|드라마|예능|방송',
                r'음악|노래|공연|춤',
                r'연예인|아이돌|배우|가수'
            ],
            "weight": 1.2
        },
        "게임": {
            "patterns": [
                r'게임|플레이|캐릭터|렙업',
                r'롤|배그|로아|메이플',
                r'겜|킬|클리어|미션'
            ],
            "weight": 1.1
        },
        "음식": {
            "patterns": [
                r'맛있|먹|음식|요리',
                r'식당|카페|메뉴|배달',
                r'점심|저녁|술|디저트'
            ],
            "weight": 1.0
        },
        "운동/건강": {
            "patterns": [
                r'운동|헬스|요가|필라',
                r'다이어트|건강|영양|식단',
                r'근육|체중|스트레칭|산책'
            ],
            "weight": 1.1
        },
        "여행": {
            "patterns": [
                r'여행|여행지|관광|투어',
                r'해외|국내|비행기|호텔',
                r'풍경|사진|구경|방문'
            ],
            "weight": 1.1
        },
        "문화생활": {
            "patterns": [
                r'전시|공연|뮤지컬|연극',
                r'영화관|공연장|미술관|박물관',
                r'책|독서|소설|만화'
            ],
            "weight": 1.0
        },
        "쇼핑": {
            "patterns": [
                r'쇼핑|구매|지름|할인',
                r'브랜드|제품|상품|아이템',
                r'옷|가방|신발|악세서리'
            ],
            "weight": 1.0
        },
        "일/공부": {
            "patterns": [
                r'일|업무|회사|프로젝트',
                r'공부|학습|시험|자격증',
                r'과제|문제|수업|강의'
            ],
            "weight": 0.9
        }
    }

    msg_text = ' '.join(messages.astype(str))
    interest_scores = {}
    
    for category, info in interest_categories.items():
        score = 0
        for pattern_group in info["patterns"]:
            matches = sum(len(re.findall(p, msg_text, re.IGNORECASE)) 
                         for p in pattern_group.split('|'))
            score += matches * info["weight"]
        interest_scores[category] = round(min(100, score * 5), 1)

    # Top 3 관심사 추출
    top_interests = dict(sorted(interest_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:3])

    return {
        "all_interests": interest_scores,
        "top_interests": top_interests
    }

def calculate_interest_score(messages: list) -> float:
    """관심사 집중도 점수 계산"""
    interest_patterns = {
        '취미': r'좋아|재미|최고|대박|짱|멋지',
        '열정': r'진짜|완전|너무|정말|매우|엄청',
        '몰입': r'찾아|봤|알아|배우|해봤|보니까',
        '지식': r'아는|알려|설명|추천|정보|후기',
        '공유': r'봐봐|들어봐|알려줄게|추천해|공유'
    }
    return calculate_average_pattern_score(messages, interest_patterns)

def analyze_message_patterns(messages: pd.Series) -> dict:
    """메시지 패턴 분석"""
    try:
        if not isinstance(messages, pd.Series):
            messages = pd.Series(messages)
            
        msg_text = ' '.join(messages.dropna().astype(str))
        
        # 관심사 패턴 정의
        interest_patterns = {
            '엔터': [r'영화|드라마|예능|방송|음악|공연|콘서트'],
            '음식': [r'맛있|먹|식당|카페|디저트|맛집|배달'],
            '게임': [r'게임|플레이|캐릭터|아이템|렙업|레벨'],
            '운동': [r'운동|헬스|요가|필라|산책|러닝|걷기'],
            '여행': [r'여행|관광|숙소|호텔|비행|휴가|놀러'],
            '쇼핑': [r'쇼핑|구매|주문|배송|상품|할인|지름'],
            '공부': [r'공부|학습|강의|수업|시험|과제|자격']
        }
        
        # 각 관심사별 점수 계산
        scores = {}
        for topic, patterns in interest_patterns.items():
            pattern_count = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, msg_text, re.IGNORECASE))
                pattern_count += matches
            
            # 메시지 길이로 정규화
            normalized_score = min(100, (pattern_count / max(1, len(messages))) * 100)
            if normalized_score > 0:  # 점수가 0인 경우는 제외
                scores[topic] = round(normalized_score, 1)
        
        # 대화 스타일 분석
        style_patterns = {
            '이모티콘': r'[😊🤣😂😍🥰😘😅❤️]',
            '감정표현': r'[ㅋㅎㅠㅜ]{2,}',
            '강조표현': r'[!?]{2,}',
            '친근표현': r'ㅇㅇ|ㄴㄴ|ㅇㅋ|ㄹㅇ|ㅊㅊ'
        }
        
        style_scores = {}
        for style, pattern in style_patterns.items():
            matches = sum(len(re.findall(pattern, str(msg), re.IGNORECASE)) for msg in messages)
            normalized_score = min(100, (matches / max(1, len(messages))) * 100)
            if normalized_score > 0:
                style_scores[style] = round(normalized_score, 1)
        
        return {
            'interests': scores,
            'style': style_scores
        }
        
    except Exception as e:
        st.error(f"메시지 패턴 분석 중 오류 발생: {str(e)}")
        return {'interests': {}, 'style': {}}


def find_dominant_patterns(results: dict) -> list:
    """가장 두드러진 패턴 찾기"""
    dominant = []
    for category, scores in results.items():
        if category != "요약":  # 요약 카테고리 제외
            max_pattern = max(scores.items(), key=lambda x: x[1])
            if max_pattern[1] > 50:  # 50점 이상인 경우만
                dominant.append({
                    "category": category,
                    "pattern": max_pattern[0],
                    "score": max_pattern[1]
                })
    
    return sorted(dominant, key=lambda x: x['score'], reverse=True)[:3]

def analyze_conversation_tendency(results: dict) -> str:
    """대화 성향 분석"""
    소통_점수 = sum(results.get('소통_방식', {}).values()) / 4
    감정_점수 = sum(results.get('감정_표현', {}).values()) / 4
    
    if 소통_점수 > 70 and 감정_점수 > 70:
        return "적극적 감성형"
    elif 소통_점수 > 70:
        return "적극적 소통형"
    elif 감정_점수 > 70:
        return "감성 표현형"
    elif 소통_점수 > 50 and 감정_점수 > 50:
        return "균형적 소통형"
    else:
        return "차분한 소통형"

def calculate_communication_level(results: dict) -> dict:
    """소통 레벨 계산"""
    # 각 카테고리별 평균 점수 계산
    category_scores = {
        category: sum(scores.values()) / len(scores)
        for category, scores in results.items()
        if category != "요약"
    }
    
    # 종합 점수 계산
    total_score = sum(category_scores.values()) / len(category_scores)
    
    # 레벨 결정
    if total_score > 80:
        level = "소통 마스터"
    elif total_score > 60:
        level = "소통 고수"
    elif total_score > 40:
        level = "소통 중수"
    else:
        level = "소통 초보"
    
    return {
        "level": level,
        "score": round(total_score, 1)
    }


def create_personality_radar_chart(metrics: dict) -> go.Figure:
    """성격 분석 레이더 차트"""
    if not metrics:
        return None
        
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # 귀여운 이모지와 함께 표시
    emoji_mapping = {
        "매력도": "💝 매력도",
        "친화력": "🤝 친화력",
        "활발도": "⚡ 활발도",
        "감성력": "💖 감성력",
        "지적호기심": "🔍 지적호기심"
    }
    
    emoji_categories = [emoji_mapping.get(cat, cat) for cat in categories]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emoji_categories,
        fill='toself',
        marker=dict(color='rgba(255, 105, 180, 0.7)'),
        line=dict(color='rgb(255, 105, 180)'),
        name='성격 특성'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white', size=12),
                ticksuffix='%',
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='white', size=14),
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        margin=dict(t=30, b=30)
    )
    
    return fig

def display_stat_metrics(title: str, metrics: dict, is_percentage: bool = False):
    """통계 메트릭을 깔끔하게 표시하는 헬퍼 함수"""
    st.markdown(f"**{title}**")
    cols = st.columns(2)
    for idx, (label, value) in enumerate(metrics.items()):
        with cols[idx % 2]:
            container = st.container()
            container.markdown(
                f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 5px 0;
                    text-align: center;
                ">
                    <div style="color: rgba(255, 255, 255, 0.7); font-size: 14px;">
                        {label}
                    </div>
                    <div style="color: #FF69B4; font-size: 20px; font-weight: bold;">
                        {value}{"%"if is_percentage and isinstance(value, (int, float)) else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def display_personality_analysis(df: pd.DataFrame, target_names: list):
    """성격 분석 결과 표시"""
    try:
        st.markdown("## 🎭 성격 분석")
        
        # 기본 통계 계산
        total_messages = len(df)
        analysis_date = df['timestamp'].max().strftime("%Y년 %m월 %d일")
        
        st.markdown(f"""
        <div style='text-align: right; color: rgba(255,255,255,0.6); margin-bottom: 20px;'>
            분석 기준일: {analysis_date}<br>
            총 분석 메시지: {total_messages:,}개
        </div>
        """, unsafe_allow_html=True)

        for name in target_names:
            with st.spinner(f"{name}님의 성격 분석 중..."):
                analysis = analyze_personality_with_gpt(df, name)
                user_msgs = df[df['name'] == name]
                msg_count = len(user_msgs)
                avg_length = user_msgs['message'].str.len().mean()
                
                # 사용자 카드 컨테이너
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(45, 45, 45, 0.7);
                        border-radius: 15px;
                        padding: 20px;
                        margin: 20px 0;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <div style="
                            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                            margin-bottom: 15px;
                            padding-bottom: 10px;
                        ">
                            <div style="color: #FF69B4; font-size: 24px; font-weight: bold;">
                                👤 {name}
                            </div>
                            <div style="color: rgba(255, 255, 255, 0.7); font-size: 16px;">
                                메시지 {msg_count:,}개 | 평균 {avg_length:.1f}자
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 주요 분석 컬럼
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        with st.container():
                            st.markdown("### 🎯 AI 성격 분석")
                            if analysis and "gpt_analysis" in analysis:
                                st.markdown(analysis["gpt_analysis"])
                            else:
                                st.error(f"{name}님의 성격 분석에 실패했습니다.")

                    with col2:
                        with st.container():
                            st.markdown("### ✨ 성격 특성 점수")
                            if analysis and "patterns" in analysis:
                                metrics = analysis.get("metrics", {})
                                if metrics:
                                    st.plotly_chart(
                                        create_personality_radar_chart(metrics),
                                        use_container_width=True
                                    )
                                    # 성격 특성 점수 표시
                                    display_stat_metrics("", metrics)
                                else:
                                    st.info("성격 특성 점수를 계산할 수 없습니다.")

                    # 추가 분석 컬럼
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        with st.container():
                            st.markdown("### 💬 대화 패턴")
                            response_times = calculate_response_patterns(df, name)
                            if response_times:
                                # 응답 시간 통계
                                response_stats = {
                                    k: v for k, v in response_times.items() 
                                    if k != "활성_시간대" and isinstance(v, (int, float))
                                }
                                display_stat_metrics("응답 패턴", response_stats, True)
                                
                                # 활성 시간대 표시
                                st.markdown("**활동 시간대**")
                                st.markdown(f"""
                                <div style="
                                    background: rgba(255, 105, 180, 0.1);
                                    border-left: 3px solid #FF69B4;
                                    padding: 10px;
                                    margin: 10px 0;
                                    border-radius: 0 8px 8px 0;
                                    font-size: 16px;
                                    color: #FF69B4;
                                ">
                                    {response_times['활성_시간대']}
                                </div>
                                """, unsafe_allow_html=True)

                    with col4:
                        with st.container():
                            st.markdown("### 🎯 관심사 & 대화 스타일")
                            patterns = analyze_message_patterns(user_msgs['message'])
                            
                            if patterns['interests']:
                                sorted_interests = dict(sorted(
                                    patterns['interests'].items(), 
                                    key=lambda x: x[1], 
                                    reverse=True
                                )[:4])
                                display_stat_metrics("주요 관심사", sorted_interests, True)
                            
                            if patterns['style']:
                                display_stat_metrics("대화 스타일", patterns['style'], True)

                    st.markdown("<hr>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"성격 분석 표시 중 오류 발생: {str(e)}")
        print(f"Display error: {str(e)}")


def calculate_response_patterns(df: pd.DataFrame, name: str) -> dict:
    """대화 응답 패턴 상세 분석"""
    try:
        # 기본 데이터 준비
        user_df = df[df['name'] == name].copy()
        df_sorted = df.sort_values('timestamp')
        
        # 시간 간격 계산
        df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds()
        
        # 1. 응답 시간 분석
        user_responses = df_sorted[
            (df_sorted['name'] == name) & 
            (df_sorted['time_diff'].notna()) & 
            (df_sorted['time_diff'] > 0) & 
            (df_sorted['time_diff'] <= 3600)  # 1시간 이내 응답만
        ]
        
        # 빠른 응답 (1분 이내)
        quick_responses = user_responses[user_responses['time_diff'] <= 60]
        # 보통 응답 (1분~5분)
        normal_responses = user_responses[(user_responses['time_diff'] > 60) & (user_responses['time_diff'] <= 300)]
        # 느린 응답 (5분~1시간)
        slow_responses = user_responses[user_responses['time_diff'] > 300]
        
        total_valid_responses = len(user_responses)
        if total_valid_responses == 0:
            return {
                "평균_응답(분)": 0,
                "빠른_응답": 0,
                "보통_응답": 0,
                "느린_응답": 0,
                "시간당_메시지": 0,
                "활성_시간대": "없음"
            }
        
        # 2. 시간대별 활동 분석
        hour_dist = user_df['timestamp'].dt.hour.value_counts()
        peak_hours = hour_dist[hour_dist >= hour_dist.mean()].index.tolist()
        peak_hours.sort()
        
        # 3. 메시지 빈도 분석
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        msgs_per_day = len(user_df) / total_days
        
        # 활성 시간대 문자열 생성
        if peak_hours:
            time_ranges = []
            start = peak_hours[0]
            prev = start
            
            for hour in peak_hours[1:] + [peak_hours[0] + 24]:
                if hour != prev + 1:
                    end = prev
                    time_ranges.append(f"{start:02d}~{end:02d}시")
                    start = hour
                prev = hour
            
            active_hours = ", ".join(time_ranges)
        else:
            active_hours = "불규칙"
        
        return {
            "평균_응답(분)": round(user_responses['time_diff'].mean() / 60, 1),
            "빠른_응답": round(len(quick_responses) / total_valid_responses * 100, 1),
            "보통_응답": round(len(normal_responses) / total_valid_responses * 100, 1),
            "느린_응답": round(len(slow_responses) / total_valid_responses * 100, 1),
            "시간당_메시지": round(msgs_per_day / 24, 1),
            "활성_시간대": active_hours
        }
        
    except Exception as e:
        st.error(f"응답 패턴 분석 중 오류 발생: {str(e)}")
        return {
            "평균_응답(분)": 0,
            "빠른_응답": 0,
            "보통_응답": 0,
            "느린_응답": 0,
            "시간당_메시지": 0,
            "활성_시간대": "분석 실패"
        }




def create_personality_radar_chart(metrics: dict) -> go.Figure:
    """성격 분석 레이더 차트"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # 귀여운 이모지와 함께 표시
    emoji_mapping = {
        "매력도": "💝 매력도",
        "친화력": "🤝 친화력",
        "활발도": "⚡ 활발도",
        "감성력": "💖 감성력",
        "지적호기심": "🔍 지적호기심"
    }
    
    emoji_categories = [emoji_mapping.get(cat, cat) for cat in categories]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emoji_categories,
        fill='toself',
        marker=dict(color='rgba(255, 105, 180, 0.7)'),
        line=dict(color='rgb(255, 105, 180)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white', size=12),
                ticksuffix='%',
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='white', size=14),
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        margin=dict(t=30, b=30)
    )
    
    return fig


def analyze_response_patterns(df: pd.DataFrame, name: str) -> dict:
    """대화 응답 패턴 분석"""
    try:
        # 기본 데이터 준비
        df = df.sort_values('timestamp')
        user_msgs = df[df['name'] == name]
        
        if len(user_msgs) == 0:
            return {"error": "메시지가 없습니다."}

        # 시간대별 활동 분석
        hour_dist = user_msgs['timestamp'].dt.hour.value_counts()
        peak_hours = hour_dist[hour_dist >= hour_dist.mean()].index.tolist()
        
        # 응답 시간 분석
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        user_responses = df[
            (df['name'] == name) & 
            (df['time_diff'].notna()) & 
            (df['time_diff'] > 0) & 
            (df['time_diff'] <= 3600)  # 1시간 이내 응답만
        ]
        
        # 빠른 응답 (1분 이내)
        quick_responses = user_responses[user_responses['time_diff'] <= 60]
        # 적당한 응답 (1분~5분)
        medium_responses = user_responses[(user_responses['time_diff'] > 60) & (user_responses['time_diff'] <= 300)]
        # 느린 응답 (5분~1시간)
        slow_responses = user_responses[user_responses['time_diff'] > 300]

        # 연속 대화 분석
        consecutive_msgs = 0
        max_consecutive = 0
        current_consecutive = 0
        prev_name = None
        
        for name_current in df['name']:
            if name_current == name:
                if prev_name == name:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
                consecutive_msgs += 1
            prev_name = name_current

        # 대화 시작 비율 (첫 발화)
        conversation_gaps = df['time_diff'] > 1800  # 30분 이상 간격을 새 대화로 간주
        new_conversations = df[conversation_gaps].index
        conversation_starts = sum(1 for idx in new_conversations if df.loc[idx, 'name'] == name)

        # 메시지 길이 패턴
        msg_lengths = user_msgs['message'].str.len()
        
        response_patterns = {
            "응답_속도": {
                "빠른_응답_비율": round(len(quick_responses) / max(1, len(user_responses)) * 100, 1),
                "평균_응답_시간": round(user_responses['time_diff'].mean() / 60, 1),  # 분 단위
                "1분내_응답": len(quick_responses),
                "5분내_응답": len(medium_responses),
                "1시간내_응답": len(slow_responses)
            },
            "대화_패턴": {
                "주요_활동_시간": sorted(peak_hours),
                "일평균_메시지": round(len(user_msgs) / df['timestamp'].dt.date.nunique(), 1),
                "연속_발화_최대": max_consecutive,
                "대화_시작_횟수": conversation_starts
            },
            "메시지_길이": {
                "평균_길이": round(msg_lengths.mean(), 1),
                "최대_길이": int(msg_lengths.max()),
                "짧은_메시지_비율": round(sum(msg_lengths < 10) / len(msg_lengths) * 100, 1),
                "긴_메시지_비율": round(sum(msg_lengths > 50) / len(msg_lengths) * 100, 1)
            },
            "대화_특성": analyze_conversation_characteristics(df, name)
        }

        return response_patterns
        
    except Exception as e:
        return {"error": str(e)}

def analyze_conversation_characteristics(df: pd.DataFrame, name: str) -> dict:
    """상세한 대화 특성 분석"""
    user_msgs = df[df['name'] == name]['message'].astype(str)
    
    # 대화 특성 패턴
    patterns = {
        "질문_하기": r'\?|어때|할까|뭐|언제|어디|누구',
        "감정_표현": r'[ㅋㅎㅠㅜ]{2,}|[😊😄😢😭😡❤️👍]',
        "동의_표현": r'그래|맞아|응|ㅇㅇ|알겠|당연',
        "존칭_사용": r'요|니다|세요|시|께서|드립',
        "강조_표현": r'!+|진짜|대박|완전|너무|정말',
        "링크_공유": r'http|www|com|net|kr|youtube',
        "이모티콘": r'\[이모티콘\]|😊|😄|😢|😭|😡|❤️|👍'
    }
    
    characteristics = {}
    total_msgs = len(user_msgs)
    
    for name, pattern in patterns.items():
        matches = sum(1 for msg in user_msgs if re.search(pattern, msg))
        ratio = round((matches / total_msgs) * 100, 1)
        characteristics[name] = ratio
        
    # 대화 스타일 결정
    style_score = {
        "친근도": characteristics.get("감정_표현", 0) + characteristics.get("이모티콘", 0),
        "적극성": characteristics.get("질문_하기", 0) + characteristics.get("강조_표현", 0),
        "공손도": characteristics.get("존칭_사용", 0),
        "정보성": characteristics.get("링크_공유", 0),
        "반응성": characteristics.get("동의_표현", 0)
    }
    
    main_style = max(style_score.items(), key=lambda x: x[1])
    
    return {
        "주요_특성": {k: v for k, v in characteristics.items() if v > 20},
        "대화_스타일": main_style[0],
        "스타일_점수": round(main_style[1], 1)
    }

def analyze_reaction_speed(df: pd.DataFrame, name: str) -> dict:
    """반응 속도 상세 분석"""
    df = df.sort_values('timestamp')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    user_responses = df[
        (df['name'] == name) & 
        (df['time_diff'].notna()) & 
        (df['time_diff'] > 0)
    ]
    
    response_times = user_responses['time_diff'].values
    
    if len(response_times) == 0:
        return {
            "error": "응답 시간을 분석할 수 없습니다."
        }
    
    speed_categories = {
        "즉각_응답": sum(response_times <= 30),  # 30초 이내
        "빠른_응답": sum((response_times > 30) & (response_times <= 60)),  # 1분 이내
        "보통_응답": sum((response_times > 60) & (response_times <= 300)),  # 5분 이내
        "느린_응답": sum((response_times > 300) & (response_times <= 3600)),  # 1시간 이내
        "매우_느린_응답": sum(response_times > 3600)  # 1시간 초과
    }
    
    total_responses = len(response_times)
    speed_ratios = {
        k: round(v / total_responses * 100, 1)
        for k, v in speed_categories.items()
    }
    
    # 응답 속도 특성 분석
    avg_speed = np.mean(response_times)
    if avg_speed <= 60:
        speed_characteristic = "매우 빠른 응답자"
    elif avg_speed <= 300:
        speed_characteristic = "신속한 응답자"
    elif avg_speed <= 900:
        speed_characteristic = "보통 응답자"
    else:
        speed_characteristic = "여유로운 응답자"
    
    return {
        "응답_분포": speed_ratios,
        "평균_응답_시간": round(avg_speed / 60, 1),  # 분 단위
        "응답_특성": speed_characteristic,
        "최소_응답_시간": round(np.min(response_times), 1),
        "최대_응답_시간": round(np.min(response_times) / 60, 1)  # 분 단위
    }

def calculate_charm_score(messages: list) -> float:
    """매력도 점수 계산"""
    charm_patterns = {
        '센스': {
            'patterns': [
                r'ㅋㅋ|ㅎㅎ|웃긴|재미|신기|멋지',
                r'굿|좋아|최고|대박|오|와',
                r'ㄷㄷ|오우|wow|헐|미쳤|실화'
            ],
            'weight': 1.2
        },
        '배려': {
            'patterns': [
                r'괜찮|도와|함께|부탁|확인',
                r'미안|감사|고마워|죄송|말씀',
                r'조심|건강|걱정|챙겨|보살'
            ],
            'weight': 1.3
        },
        '적극성': {
            'patterns': [
                r'하자|가자|좋지|당연|오케이',
                r'그래|해보|진행|시도|도전',
                r'완전|진짜|너무|정말|매우'
            ],
            'weight': 1.1
        },
        '감성': {
            'patterns': [
                r'예쁘|멋지|귀엽|사랑|좋아',
                r'행복|그립|보고싶|아름|설레',
                r'❤️|💕|😊|🥰|😍'
            ],
            'weight': 1.0
        },
        '유머': {
            'patterns': [
                r'ㅋㅋㅋ|ㅎㅎㅎ|웃겨|재밌|웃김',
                r'드립|개그|농담|장난|유머',
                r'킹|짱|대박|역대급|레전드'
            ],
            'weight': 1.2
        }
    }
    
    return calculate_weighted_pattern_score(messages, charm_patterns)

def calculate_friendship_score(messages: list) -> float:
    """친화력 점수 계산"""
    friendship_patterns = {
        '공감': {
            'patterns': [
                r'맞아|그래|이해|당연|심하',
                r'같은|비슷|역시|그쵸|동감',
                r'나도|저도|우리도|마찬|처럼'
            ],
            'weight': 1.3
        },
        '관심': {
            'patterns': [
                r'어때|괜찮|요즘|근데|무슨',
                r'왜|뭐|누구|언제|어디',
                r'혹시|있어|없어|같은|보니'
            ],
            'weight': 1.1
        },
        '지지': {
            'patterns': [
                r'힘내|화이팅|응원|좋겠|기대',
                r'대단|축하|수고|고생|잘했',
                r'믿어|아자|파이팅|굿|최고'
            ],
            'weight': 1.2
        },
        '친근': {
            'patterns': [
                r'야|여기|저기|너|나',
                r'우리|같이|둘이|다같|함께',
                r'친구|동생|언니|오빠|형'
            ],
            'weight': 1.0
        }
    }
    
    return calculate_weighted_pattern_score(messages, friendship_patterns)

def calculate_wit_score(messages: list) -> float:
    """재치력 점수 계산"""
    wit_patterns = {
        '순발력': {
            'patterns': [
                r'바로|당연|역시|쿨|굿|오케',
                r'즉시|빨리|금방|순식|재빨',
                r'척척|번개|순간|즉각|바람'
            ],
            'weight': 1.2
        },
        '센스': {
            'patterns': [
                r'센스|깔끔|딱|찰떡|완벽',
                r'기가|갓|뇌섹|천재|현실',
                r'예리|똑똑|영리|명쾌|탁월'
            ],
            'weight': 1.3
        },
        '유머': {
            'patterns': [
                r'ㅋㅋ|웃긴|재미|드립|웃음',
                r'장난|농담|개그|코미|짖궃',
                r'티키|타키|꾸르|잼민|레전'
            ],
            'weight': 1.1
        }
    }
    
    return calculate_weighted_pattern_score(messages, wit_patterns)

def calculate_emotion_score(messages: list) -> float:
    """감성력 점수 계산"""
    emotion_patterns = {
        '감정표현': {
            'patterns': [
                r'좋아|행복|기쁘|슬프|화나',
                r'그립|설레|궁금|답답|흐뭇',
                r'😊|😢|😭|😡|🥰'
            ],
            'weight': 1.3
        },
        '공감': {
            'patterns': [
                r'이해|맞아|그렇|당연|같이',
                r'느낌|서로|우리|함께|공감',
                r'아니|역시|진짜|정말|너무'
            ],
            'weight': 1.2
        },
        '배려': {
            'patterns': [
                r'괜찮|미안|감사|부탁|걱정',
                r'조심|건강|힘내|위로|도와',
                r'주의|보살|챙기|신경|배려'
            ],
            'weight': 1.1
        }
    }
    
    return calculate_weighted_pattern_score(messages, emotion_patterns)

def calculate_interest_score(messages: list) -> float:
    """관심사 집중도 점수 계산"""
    interest_patterns = {
        '취미': {
            'patterns': [
                r'좋아|재미|최고|대박|짱',
                r'취미|관심|열정|몰입|빠져',
                r'즐기|놀기|놀러|체험|경험'
            ],
            'weight': 1.2
        },
        '학습': {
            'patterns': [
                r'공부|배우|연구|학습|강의',
                r'책|독서|자격|시험|과정',
                r'정보|지식|이론|실전|실습'
            ],
            'weight': 1.1
        },
        '문화': {
            'patterns': [
                r'영화|드라마|음악|공연|전시',
                r'게임|만화|웹툰|소설|책',
                r'예술|문화|공연|작품|감상'
            ],
            'weight': 1.0
        }
    }
    
    return calculate_weighted_pattern_score(messages, interest_patterns)

def calculate_weighted_pattern_score(messages: list, pattern_categories: dict) -> float:
    """가중치가 적용된 패턴 점수 계산"""
    if not messages:
        return 0
    
    msg_text = ' '.join(str(m) for m in messages if isinstance(m, str))
    total_score = 0
    total_weight = 0
    
    for category, info in pattern_categories.items():
        category_score = 0
        patterns = info['patterns']
        weight = info['weight']
        
        for pattern_group in patterns:
            matches = sum(len(re.findall(p, msg_text, re.IGNORECASE)) 
                         for p in pattern_group.split('|'))
            # 패턴 그룹당 최대 점수 제한
            category_score += min(100, matches * 5)
        
        # 카테고리 평균 점수 계산
        avg_category_score = category_score / len(patterns)
        total_score += avg_category_score * weight
        total_weight += weight
    
    # 최종 점수 계산 (0-100 범위로 정규화)
    final_score = (total_score / total_weight) if total_weight > 0 else 0
    return round(min(100, final_score), 1)


def main():
    st.title("💬 카톡 대화 분석기")
    st.markdown("### AI가 여러분의 카톡방을 분석해드려요! 🤖")

    with st.sidebar:
        st.markdown("""
        ### 📱 사용 방법
        1. 카카오톡 대화창 → 메뉴
        2. 대화내용 내보내기 (.txt)
        3. 파일 업로드
        4. 분석 시작!
        
        ### 🔍 분석 항목
        - 대화 맥락과 관계
        - 주제별 분석
        - 감정 분석
        - 대화 패턴
        - GPT 심층 분석
        - 성격 분석
        """)

    # 파일 업로드
    chat_file = st.file_uploader(
        "카카오톡 대화 내보내기 파일을 올려주세요 (.txt)",
        type=['txt']
    )
    
    if chat_file:
        with st.spinner("대화 내용을 불러오는 중..."):
            chat_text = chat_file.read().decode('utf-8')
            df = parse_kakao_chat(chat_text)
        
        if len(df) > 0:
            unique_names = df['name'].unique()
            
            # 참여자 선택
            col1, col2 = st.columns(2)
            with col1:
                my_name = st.selectbox(
                    "분석할 대화방의 당신 이름을 선택하세요",
                    options=unique_names
                )
            with col2:
                target_names = st.multiselect(
                    "분석할 대상을 선택하세요 (여러 명 선택 가능)",
                    options=[n for n in unique_names if n != my_name],
                    default=[n for n in unique_names if n != my_name]
                )
            
            if st.button("🔍 대화 분석 시작", use_container_width=True):
                with st.spinner("AI가 대화를 심층 분석중입니다..."):
                    # 1. 기본 통계
                    st.markdown("## 📊 대화방 기본 통계")
                    date_range = (df['timestamp'].max() - df['timestamp'].min())
                    total_duration = date_range.days + 1  # 최소 1일
                    unique_dates = len(df['timestamp'].dt.date.unique())
                    total_messages = len(df)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "총 대화 기간",
                            f"{unique_dates}일",
                            f"전체 {total_duration}일 중"
                        )
                    with col2:
                        daily_avg = total_messages / unique_dates if unique_dates > 0 else total_messages
                        st.metric(
                            "총 메시지 수",
                            f"{total_messages:,}개",
                            f"하루 평균 {daily_avg:.1f}개"
                        )
                    with col3:
                        active_users = len([n for n in unique_names 
                                        if len(df[df['name']==n]) > total_messages*0.1])
                        st.metric(
                            "참여자 수",
                            f"{len(unique_names)}명",
                            f"활성 사용자 {active_users}명"
                        )
                    
                    # 2. GPT 대화 분석
                    st.markdown("## 🤖 AI 대화 분석")
                    analysis = analyze_chat_context(df, target_names, my_name)
                    
                    if analysis:
                        tabs = st.tabs(["💡 전반적 분석", "👥 관계 분석", "📊 주제 분석", "📈 패턴 분석"])
                        
                        with tabs[0]:
                            if 'gpt_analysis' in analysis:
                                st.markdown(analysis['gpt_analysis'])
                            else:
                                st.info("GPT 분석 결과를 불러올 수 없습니다.")
                        
                        with tabs[1]:
                            st.markdown("### 참여자 관계도")
                            if 'relationships' in analysis:
                                st.plotly_chart(
                                    create_relationship_graph(analysis['relationships']),
                                    use_container_width=True
                                )
                            else:
                                st.info("관계 분석 데이터를 불러올 수 없습니다.")
                        
                        with tabs[2]:
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'topics' in analysis:
                                    st.plotly_chart(
                                        create_topic_chart(analysis['topics']),
                                        use_container_width=True
                                    )
                                else:
                                    st.info("주제 분석 데이터를 불러올 수 없습니다.")
                            with col2:
                                st.markdown("### 주요 키워드")
                                wordcloud_fig = create_wordcloud(df['message'])
                                if wordcloud_fig:
                                    st.pyplot(wordcloud_fig)
                                else:
                                    st.info("워드클라우드를 생성할 수 없습니다.")
                        
                        with tabs[3]:
                            st.markdown("### 시간대별 대화 패턴")
                            st.plotly_chart(
                                create_time_pattern(df, target_names, my_name),
                                use_container_width=True
                            )
                    
                    # 3. 성격 분석 (새로 추가된 섹션)
                    st.markdown("## 🎭 성격 분석")
                    display_personality_analysis(df, target_names)
                    
                    # 4. 상세 분석
                    st.markdown("## 📱 상세 대화 분석")
                    
                    # 대화량 분석
                    st.markdown("### 💬 참여자별 대화량")
                    conversation_stats = analyze_conversation_stats(df)
                    st.plotly_chart(
                        create_conversation_chart(conversation_stats),
                        use_container_width=True
                    )
                    
                    # 감정 분석
                    st.markdown("### 😊 감정 분석")
                    col1, col2 = st.columns(2)
                    with col1:
                        emotion_stats = analyze_emotions(df)
                        st.plotly_chart(
                            create_emotion_chart(emotion_stats),
                            use_container_width=True
                        )
                    with col2:
                        st.markdown("#### 주요 감정 키워드")
                        try:
                            emotion_cloud = create_detailed_wordcloud(df['message'])
                            if emotion_cloud:
                                st.pyplot(emotion_cloud)
                            else:
                                st.info("워드클라우드를 생성할 수 없습니다. 대체 분석을 표시합니다.")
                                # 대체 분석: 상위 감정 키워드 표시
                                emotions = df['message'].str.findall(r'[ㅋㅎ]{2,}|[ㅠㅜ]{2,}|[!?]{2,}|😊|😄|😢|😭|😡|❤️|👍|🙏').explode()
                                top_emotions = emotions.value_counts().head(10)
                                if not top_emotions.empty:
                                    st.write("가장 많이 사용된 감정 표현:")
                                    for emotion, count in top_emotions.items():
                                        st.write(f"- {emotion}: {count}회")
                                else:
                                    st.write("감정 표현을 찾을 수 없습니다.")
                        except Exception as e:
                            st.error(f"감정 분석 표시 중 오류 발생: {str(e)}")
                    
                    # 대화 하이라이트
                    st.markdown("## ✨ 대화 하이라이트")
                    highlights = find_highlight_messages(df, target_names, my_name)
                    
                    tabs = st.tabs(["💝 인상적인 대화", "🚀 활발한 토론", "⚡ 빠른 답장"])
                    with tabs[0]:
                        if highlights and 'emotional_messages' in highlights:
                            for msg in highlights['emotional_messages']:
                                st.info(f"{msg['timestamp'].strftime('%Y-%m-%d %H:%M')} - {msg['name']}: {msg['message']}")
                        else:
                            st.write("인상적인 대화를 찾을 수 없습니다.")

                    with tabs[1]:
                        if highlights and 'discussion_messages' in highlights:
                            for msg in highlights['discussion_messages']:
                                st.info(f"{msg['timestamp'].strftime('%Y-%m-%d %H:%M')} - {msg['name']}: {msg['message']}")
                        else:
                            st.write("활발한 토론을 찾을 수 없습니다.")

                    with tabs[2]:
                        if highlights and 'quick_responses' in highlights:
                            for msg in highlights['quick_responses']:
                                st.info(f"{msg['timestamp'].strftime('%Y-%m-%d %H:%M')} - {msg['name']}: {msg['message']}")
                        else:
                            st.write("빠른 답장을 찾을 수 없습니다.")
                    
                    # 5. AI 제안
                    display_suggestions(analysis)
                                        
        else:
            st.error("채팅 데이터를 파싱할 수 없습니다. 올바른 카카오톡 대화 파일인지 확인해주세요.")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by AI")

if __name__ == "__main__":
    main()