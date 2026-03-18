import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="$S_{BET}$& CO₂ Prediction Web App", page_icon="🧪", layout="centered")

st.title("$S_{BET}$ & $CO_{2}$ $Prediction Web App$")
st.caption("基于 RF 主模型与平滑代理模型，同时预测 $S_{BET}$ 和 CO₂ 吸附值")


DEFAULT_RF_SBET_MODEL_PATH = "models/final_rf_sbet_model.pkl"
DEFAULT_SMOOTH_SBET_MODEL_PATH = "models/smooth_sbet_model.pkl"
DEFAULT_RF_CO2_MODEL_PATH = "models/final_rf_co2_model.pkl"
DEFAULT_SMOOTH_CO2_MODEL_PATH = "models/smooth_co2_model.pkl"

with st.sidebar:
    st.header("模型设置")

    st.subheader("$S_{BET}$ 模型")
    rf_sbet_model_path = st.text_input("$S_{BET}$ RF模型路径", value=DEFAULT_RF_SBET_MODEL_PATH)
    smooth_sbet_model_path = st.text_input("$S_{BET}$ 平滑模型路径", value=DEFAULT_SMOOTH_SBET_MODEL_PATH)

    st.subheader("CO₂ 模型")
    rf_co2_model_path = st.text_input("CO₂ RF模型路径", value=DEFAULT_RF_CO2_MODEL_PATH)
    smooth_co2_model_path = st.text_input("CO₂ 平滑模型路径", value=DEFAULT_SMOOTH_CO2_MODEL_PATH)


@st.cache_resource
def load_four_models(
    rf_sbet_path,
    smooth_sbet_path,
    rf_co2_path,
    smooth_co2_path
):

    rf_sbet_loaded = joblib.load(rf_sbet_path)
    rf_sbet_model = rf_sbet_loaded["model"]
    rf_sbet_scaler = rf_sbet_loaded["scaler"]
    rf_sbet_features = rf_sbet_loaded["features"]


    smooth_sbet_loaded = joblib.load(smooth_sbet_path)
    smooth_sbet_model = smooth_sbet_loaded["model"]
    smooth_sbet_features = smooth_sbet_loaded["features"]


    rf_co2_loaded = joblib.load(rf_co2_path)
    rf_co2_model = rf_co2_loaded["model"]
    rf_co2_scaler = rf_co2_loaded["scaler"]
    rf_co2_features = rf_co2_loaded["features"]


    smooth_co2_loaded = joblib.load(smooth_co2_path)
    smooth_co2_model = smooth_co2_loaded["model"]
    smooth_co2_features = smooth_co2_loaded["features"]

    return {
        "rf_sbet_model": rf_sbet_model,
        "rf_sbet_scaler": rf_sbet_scaler,
        "rf_sbet_features": rf_sbet_features,
        "smooth_sbet_model": smooth_sbet_model,
        "smooth_sbet_features": smooth_sbet_features,
        "rf_co2_model": rf_co2_model,
        "rf_co2_scaler": rf_co2_scaler,
        "rf_co2_features": rf_co2_features,
        "smooth_co2_model": smooth_co2_model,
        "smooth_co2_features": smooth_co2_features
    }


try:
    models = load_four_models(
        rf_sbet_model_path,
        smooth_sbet_model_path,
        rf_co2_model_path,
        smooth_co2_model_path
    )
    st.success("四个模型加载成功")
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()


st.subheader("1. 单组实验条件预测")

col1, col2 = st.columns(2)
with col1:
    htct = st.number_input("HTCT (°C)", min_value=180.0, max_value=300.0, value=240.0, step=1.0)
    htime = st.number_input("Htime (min)", min_value=60.0, max_value=360.0, value=210.0, step=1.0)
    koh = st.number_input("KOH/hydrochar", min_value=0.0, max_value=4.0, value=4.0, step=0.1)

with col2:
    at = st.number_input("AT (°C)", min_value=600.0, max_value=900.0, value=750.0, step=1.0)
    atime = st.number_input("Atime (min)", min_value=30.0, max_value=180.0, value=105.0, step=1.0)

sample = pd.DataFrame([{
    "HTCT": htct,
    "Htime": htime,
    "KOH/hydrochar": koh,
    "AT": at,
    "Atime": atime
}])

if st.button("开始预测", type="primary"):



    sbet_rf_features = models["rf_sbet_features"]
    sbet_rf_scaler = models["rf_sbet_scaler"]
    sbet_rf_model = models["rf_sbet_model"]
    sbet_smooth_model = models["smooth_sbet_model"]
    sbet_smooth_features = models["smooth_sbet_features"]

    sample_sbet_scaled = sbet_rf_scaler.transform(sample[sbet_rf_features])
    rf_sbet_pred = sbet_rf_model.predict(sample_sbet_scaled)[0]
    smooth_sbet_pred = sbet_smooth_model.predict(sample[sbet_smooth_features])[0]



    co2_rf_features = models["rf_co2_features"]
    co2_rf_scaler = models["rf_co2_scaler"]
    co2_rf_model = models["rf_co2_model"]
    co2_smooth_model = models["smooth_co2_model"]
    co2_smooth_features = models["smooth_co2_features"]

    sample_co2_scaled = co2_rf_scaler.transform(sample[co2_rf_features])
    rf_co2_pred = co2_rf_model.predict(sample_co2_scaled)[0]
    smooth_co2_pred = co2_smooth_model.predict(sample[co2_smooth_features])[0]



    st.markdown("### 2. 预测结果")

    r1, r2 = st.columns(2)
    with r1:
        st.success(f"SBET - RF预测值 = {rf_sbet_pred:.5f}")
        st.info(f"SBET - 平滑预测值 = {smooth_sbet_pred:.5f}")
    with r2:
        st.success(f"CO₂ - RF预测值 = {rf_co2_pred:.5f}")
        st.info(f"CO₂ - 平滑预测值 = {smooth_co2_pred:.5f}")

    st.markdown("### 3. 输入实验条件")
    st.dataframe(sample, use_container_width=True)



st.divider()
st.subheader("4. 批量实验条件预测")
st.write("上传 CSV 文件，列名需为：HTCT, Htime, KOH/hydrochar, AT, Atime")

uploaded_file = st.file_uploader("上传待预测 CSV 文件", type=["csv"])

if uploaded_file is not None:
    try:
        new_conditions = pd.read_csv(uploaded_file)

        required_cols = ["HTCT", "Htime", "KOH/hydrochar", "AT", "Atime"]
        missing_cols = [c for c in required_cols if c not in new_conditions.columns]

        if missing_cols:
            st.error(f"CSV 缺少必要列：{missing_cols}")
        else:

            sbet_rf_scaled = models["rf_sbet_scaler"].transform(new_conditions[models["rf_sbet_features"]])
            rf_sbet_preds = models["rf_sbet_model"].predict(sbet_rf_scaled)
            smooth_sbet_preds = models["smooth_sbet_model"].predict(new_conditions[models["smooth_sbet_features"]])


            co2_rf_scaled = models["rf_co2_scaler"].transform(new_conditions[models["rf_co2_features"]])
            rf_co2_preds = models["rf_co2_model"].predict(co2_rf_scaled)
            smooth_co2_preds = models["smooth_co2_model"].predict(new_conditions[models["smooth_co2_features"]])

            result_df = new_conditions.copy()
            result_df["RF_Predicted_SBET"] = rf_sbet_preds
            result_df["Smoothed_Predicted_$S_{BET}$"] = smooth_sbet_preds
            result_df["RF_Predicted_CO2"] = rf_co2_preds
            result_df["Smoothed_Predicted_CO2"] = smooth_co2_preds

            st.success("批量预测完成")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载预测结果 CSV",
                data=csv_data,
                file_name="predicted_sbet_co2_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"批量预测失败：{e}")



st.divider()
st.subheader("5. 说明")
st.markdown(
    """
    - **RF预测值**：原始随机森林主模型的直接输出，精度较高，但在小范围参数变化时可能保持不变。  
    - **平滑预测值**：由平滑代理模型输出，更适合连续调参与网页展示。  
    - 本网页同时支持 **SBET** 与 **CO₂ 吸附值** 的单组预测和批量预测。  
    """
)
