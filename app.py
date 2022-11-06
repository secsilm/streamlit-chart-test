import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from bokeh.plotting import figure

st.title("Stremlit Chart 测试")
st.write(
    """测试 streamlit 中不同的 [chart components](https://docs.streamlit.io/library/api-reference/charts) 的显示效果。性能上来看，plotly 最好，绘制 10 万个数据点时仍然能保证不卡，其他均有卡顿。"""
)


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf


# st.bar_chart会对x进行排序，且不可更改。该函数其实是对altair_chart的简单封装。
# st.bar_chart(df, x='country', y='distance')
num_datapoints = st.number_input("请输入数据点数量", value=1000, min_value=1)
x = np.linspace(-10, 10, num_datapoints)
y = gelu(x) + np.random.randn(num_datapoints)
df = pd.DataFrame({"x": x, "y": y})

st.write("Streamlit built-in line chart，不能设置标题。")
st.line_chart(df, x="x", y="y")

st.write("---")

chart = (
    alt.Chart(df)
    .mark_line()
    .encode(x="x", y="y",)
    .properties(title="Altair line chart")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.write("---")

chart = px.line(df, x="x", y="y", title="Plotly express line chart")
st.plotly_chart(chart)

st.write("---")

x = df.x.tolist()
y = df.y.tolist()
chart = figure(title="Bokeh line chart")
chart.line(x=x, y=y)
st.bokeh_chart(chart)
