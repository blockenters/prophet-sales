import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)



def load_data():
    # 데이터 로드
    df = pd.read_csv('data/realistic_sales_data.csv', parse_dates=['date'])
    df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    return df

def plot_sales_data(df):
    # 데이터 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ds'], df['y'], label='일일 매출')
    ax.set_title('일일 매출 데이터 시각화')
    ax.set_xlabel('날짜')
    ax.set_ylabel('매출')
    ax.legend()
    return fig

def create_holidays_df():
    # 휴일 데이터 생성
    return pd.DataFrame({
        'holiday': ['new_year', 'promotion_may', 'promotion_nov', 'black_friday'],
        'ds': pd.to_datetime(['2020-01-01', '2021-05-15', '2022-11-20', '2023-11-24']),
        'lower_window': 0,
        'upper_window': 1
    })

def train_and_predict(df, periods=90):
    # Prophet 모델 학습 및 예측
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=create_holidays_df(),
        changepoint_prior_scale=0.1
    )
    
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def analyze_trend(df):
    # 전체 트렌드 분석
    total_growth = ((df['y'].iloc[-1] - df['y'].iloc[0]) / df['y'].iloc[0] * 100).round(2)
    yearly_avg = df.groupby(df['ds'].dt.year)['y'].mean()
    yoy_growth = ((yearly_avg.iloc[-1] - yearly_avg.iloc[-2]) / yearly_avg.iloc[-2] * 100).round(2)
    
    return {
        "total_growth": total_growth,
        "yoy_growth": yoy_growth,
        "max_sales": df['y'].max(),
        "min_sales": df['y'].min(),
        "avg_sales": df['y'].mean().round(2)
    }

def analyze_seasonality(df):
    # 계절성 분석
    monthly_avg = df.groupby(df['ds'].dt.month)['y'].mean()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    
    daily_avg = df.groupby(df['ds'].dt.day_name())['y'].mean()
    best_day = daily_avg.idxmax()
    worst_day = daily_avg.idxmin()
    
    return {
        "best_month": best_month,
        "worst_month": worst_month,
        "best_day": best_day,
        "worst_day": worst_day
    }

def main():
    fontRegistered()
    plt.rc('font', family='NanumGothic')    
    

    st.title("📊 매출 예측 대시보드")
    
    st.write("📈 시계열 데이터 예측 모델을 활용하여 과거 매출 데이터를 분석하고 미래 매출을 예측하는 대시보드입니다.")
    
    st.markdown("---")  # 구분선 추가
    
    df = load_data()
    
    # 기본 데이터 분석
    trend_analysis = analyze_trend(df)
    season_analysis = analyze_seasonality(df)
    
    # 주요 지표를 상단에 배치
    st.subheader("💡 주요 매출 지표")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 전체 성장률", f"{trend_analysis['total_growth']}%")
    with col2:
        st.metric("🔄 전년 대비 성장률", f"{trend_analysis['yoy_growth']}%")
    with col3:
        st.metric("📊 평균 일일 매출", f"{trend_analysis['avg_sales']}건")
    with col4:
        st.metric("⬆️ 최고 매출", f"{trend_analysis['max_sales']}건")
    with col5:
        st.metric("⬇️ 최저 매출", f"{trend_analysis['min_sales']}건")
    
    # 매출 패턴 분석을 2행으로 배치
    st.markdown("---")
    st.subheader("📊 매출 패턴 분석")
    pattern_col1, pattern_col2 = st.columns(2)
    
    with pattern_col1:
        st.write("**📅 월별 패턴**")
        st.write(f"• 최고 매출 월: {season_analysis['best_month']}월")
        st.write(f"• 최저 매출 월: {season_analysis['worst_month']}월")
    
    with pattern_col2:
        st.write("**📆 요일별 패턴**")
        st.write(f"• 최고 매출 요일: {season_analysis['best_day']}")
        st.write(f"• 최저 매출 요일: {season_analysis['worst_day']}")
    
    # 데이터 시각화
    st.markdown("---")
    st.subheader("📈 일일 매출 데이터 분석")
    st.pyplot(plot_sales_data(df))
    
    # 예측 기간 설정
    periods = st.slider("예측 기간 (일)", min_value=30, max_value=365, value=90)
    
    if st.button("🎯 예측 시작"):
        with st.spinner("🔄 예측 중..."):
            model, forecast = train_and_predict(df, periods)
            
            # 예측 결과 시각화 및 해석
            st.subheader("🎯 매출 예측 결과")
            fig1 = model.plot(forecast)
            plt.title('Prophet 모델 기반 매출 예측')
            plt.xlabel('날짜')
            plt.ylabel('매출')
            st.pyplot(fig1)
            
            st.write("### 🔍 예측 결과 해석")
            future_trend = (forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / forecast['yhat'].iloc[0] * 100
            st.write(f"• 예측 기간 동안의 예상 성장률: {future_trend:.2f}%")
            st.write("• 파란색 선: 실제 매출 데이터")
            st.write("• 빨간색 선: 예측된 매출")
            st.write("• 파란색 영역: 예측의 불확실성 범위")
            
            # 트렌드 및 계절성 분해
            st.subheader("📊 트렌드 및 계절성 분석")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            st.write("### 📋 분해 요소 해석")
            st.write("**1. 트렌드 (Trend)**")
            st.write("- 전반적인 매출의 장기적 추세를 보여줍니다.")
            st.write("- 계절성이나 단기 변동을 제외한 순수한 성장 패턴입니다.")
            
            st.write("**2. 주간 패턴 (Weekly)**")
            st.write("- 요일별 매출 패턴을 보여줍니다.")
            st.write("- 주말과 평일의 매출 차이를 확인할 수 있습니다.")
            
            st.write("**3. 연간 패턴 (Yearly)**")
            st.write("- 1년 주기의 계절성을 보여줍니다.")
            st.write("- 특정 월이나 계절의 매출 패턴을 파악할 수 있습니다.")
            
            # 예측 결과 다운로드 및 테이블
            forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 예측 결과 다운로드",
                data=forecast_csv,
                file_name="sales_forecast_results.csv",
                mime="text/csv"
            )
            
            st.subheader("📅 향후 7일 예측 결과")
            future_forecast = forecast[forecast['ds'] > df['ds'].max()].head(7)
            st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2))
            
            st.write("### 📝 예측 결과 설명")
            st.write("- **yhat**: 예측된 매출값")
            st.write("- **yhat_lower**: 예측의 하한값 (80% 신뢰구간)")
            st.write("- **yhat_upper**: 예측의 상한값 (80% 신뢰구간)")

if __name__ == "__main__":
    main()



